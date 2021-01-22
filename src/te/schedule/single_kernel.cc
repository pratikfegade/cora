#include <tvm/ir/attrs.h>
#include <tvm/te/operation.h>

#include "graph.h"
#include "schedule_utils.h"

#define COUT std::cout << "[SK] "
namespace tvm {
namespace te {

Operation CreateSingleKernel(Schedule& sch, std::string name, std::string tag,
                             Map<std::string, ObjectRef> attrs, const Array<Tensor>& inputs_,
                             const Array<Tensor>& outputs_, bool include_inputs,
                             const Array<Dimension>& explicit_dimensions,
                             const Array<IterVar>& thread_vars) {
  CHECK((thread_vars.size() > 0) ^ (explicit_dimensions.size() > 0));
  // CheckSchedule(sch, "0", true);
  sch->InvalidateCache();
  ScheduleNode* self = sch.operator->();
  const Array<Tensor>& inputs = RemapTensor(self, inputs_);
  const Array<Tensor>& outputs = RemapTensor(self, outputs_);

  /************** Create the envelope op **************/
  Operation envelope =
      SingleKernelEnvelopeOpNode::make(name, tag, attrs, explicit_dimensions, outputs);

  /************** Replace tensors **************/
  std::unordered_map<Tensor, Tensor> vmap;
  std::unordered_map<Tensor, Tensor> rvmap;
  Array<Tensor> new_outputs;
  for (size_t i = 0; i < outputs.size(); ++i) {
    Tensor output = envelope.output(i);
    vmap[outputs[i]] = output;
    rvmap[output] = outputs[i];
    new_outputs.push_back(output);
    std::cout << "[SK] To replace tensor " << outputs[i] << " " << output << std::endl;
  }
  // std::cout << "[SK] RDF" << std::endl;
  // CheckSchedule(sch, "0.5", true);

  std::unordered_set<const OperationNode*> output_ops;
  for (const auto& t : outputs) {
    output_ops.insert(t->op.as<OperationNode>());
  }

  // std::unordered_set<const OperationNode*> to_skip_ops;
  // for (const auto& t : te::GetSubGraph(outputs, inputs, include_inputs)) {
  // to_skip_ops.insert(t.as<OperationNode>());
  // }
  ReplaceDataFlow(sch->stages, sch->cacheTensorInfos, &vmap, &rvmap, output_ops);
  Stage envelope_stage = Stage(envelope);

  CheckSchedule(sch, "1", true);

  /************** Update stages **************/
  ArrayNode* stages = sch->stages.CopyOnWrite();
  size_t pos = 0;
  for (const auto& output : outputs) {
    pos = std::max(pos, FindNodeRef(stages, self->op2stage_cache_[output->op.get()]));
  }
  CHECK_LT(pos, stages->data.size());
  stages->data.insert(stages->data.begin() + pos + 1, envelope_stage);
  sch->stage_map.Set(envelope, envelope_stage);
  envelope_stage.env_threads(thread_vars);
  // CheckSchedule(sch, "2", true);

  /************** Update schedule outputs **************/
  ArrayNode* sch_outputs = sch->outputs.CopyOnWrite();
  bool have_output = false;
  Stage output_group = NullValue<Stage>();
  for (const auto& output : outputs) {
    Stage s = self->op2stage_cache_[output->op.get()];
    if (s->group.defined()) {
      CHECK(!output_group.defined() || output_group == s->group)
          << "Outputs are in different groups";
      output_group = s->group;
    }
    if (s->is_output) {
      have_output = true;
      s->is_output = false;
      pos = FindNodeRef(sch_outputs, s->origin_op);
      CHECK(pos < sch->outputs.size());
      sch_outputs->data.erase(sch_outputs->data.begin() + pos);
    }
  }
  if (have_output) {
    envelope_stage->is_output = have_output;
    sch_outputs->data.insert(sch_outputs->data.end(), envelope);
  }

  for (auto s : sch->stages) {
    CHECK(s->attach_type != kSingleKernelScope);
  }
  // CheckSchedule(sch, "2", true);

  /************** Create group **************/
  // Create a group out of all the passed ops and attach that to the
  // stage of the envelope op
  Array<Tensor> group_outputs = Array<Tensor>(outputs);
  // for (size_t i = 0; i < outputs.size(); ++i) {
  // group_outputs.push_back(envelope.output(i));
  // }
  Stage group = sch.create_group(group_outputs, inputs, include_inputs);
  group->attach_type = kSingleKernelScope;
  if (thread_vars.size() > 0) {
    group->attach_ivar = thread_vars[thread_vars.size() - 1];
  } else {
    auto last_dim = explicit_dimensions[explicit_dimensions.size() - 1];
    CHECK(last_dim->isLoopDim()) << "Last dim should be a loop dim";
    group->attach_ivar = envelope.as<SingleKernelEnvelopeOpNode>()->GetIterVarFromDim(0, last_dim);
  }
  // group->attach_type = kScope;
  group->attach_stage = envelope_stage;

  // If all outputs are in a group, set the envelope stage to be in
  // the same group.
  envelope_stage->group = output_group;
  if (output_group.defined()) {
    ++output_group->num_child_stages;
  }

  // return output_tensors;
  // return envelope.output(0);
  // std::cout << "[SK] REt " << envelope << std::endl;
  CheckSchedule(sch, "single_kernel.cc:120_end_" + name, true);
  sch->remakePostOrder();
  return envelope;
}

Operation Schedule::unify(std::string name, std::string tag, Map<std::string, ObjectRef> attrs,
                          const Array<Tensor>& tensors,
                          const Array<Dimension>& explicit_dimensions) {
  return CreateSingleKernel(*this, name, tag, attrs, tensors, tensors, true, explicit_dimensions,
                            {});
}

Operation Schedule::single_kernel(std::string name, std::string tag,
                                  Map<std::string, ObjectRef> attrs, const Array<Tensor>& inputs_,
                                  const Array<Tensor>& outputs_, bool include_inputs,
                                  const Array<IterVar>& thread_vars) {
  return CreateSingleKernel(*this, name, tag, attrs, inputs_, outputs_, include_inputs, {},
                            thread_vars);
}
}  // namespace te
}  // namespace tvm
