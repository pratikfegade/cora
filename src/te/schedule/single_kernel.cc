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
  // CheckSchedule(sch, "0");
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
  }

  std::unordered_set<const OperationNode*> output_ops;
  // for (const auto& t : outputs) {
    // output_ops.insert(t->op.as<OperationNode>());
  // }
  ReplaceDataFlow(sch->stages, sch->cacheTensorInfos, &vmap, &rvmap, output_ops);


  SingleKernelEnvelopeOpNode* mut_op =
    const_cast<SingleKernelEnvelopeOpNode*>(envelope.as<SingleKernelEnvelopeOpNode>());

  Array<Tensor> new_inputs;
  std::vector<const BaseVarDimOpNode*> new_input_ops;
  for (auto t: mut_op->inputs) {
    if (vmap.count(t) && vmap.at(t)->op != envelope) {
      new_inputs.push_back(vmap.at(t));
      new_input_ops.push_back(vmap.at(t)->op.as<BaseVarDimOpNode>());
    } else {
      new_inputs.push_back(t);
      new_input_ops.push_back(t->op.as<BaseVarDimOpNode>());
    }
  }

  mut_op->inputs = new_inputs;
  mut_op->input_ops = new_input_ops;

  // for (auto it: vmap) {
  //   std::cout << "BALLEBALLE " << it.first->op << " " << it.second->op << std::endl;
  // }

  Stage envelope_stage = Stage(envelope);

  // CheckSchedule(sch, "0");

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

  // CheckSchedule(sch, "1");

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
      pos = FindNodeRef(sch_outputs, output->op);
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

  // std::cout << "[SK] Group stage " << group << std::endl;

  // return output_tensors;
  // return envelope.output(0);

  sch->remakePostOrder();

  // std::cout << "[UNIUNIUNIFY]" << std::endl;
  // for (size_t i = sch->stages.size(); i != 0; --i) {
  //   const Stage& stage = sch->stages[i - 1];
  //   std::cout << "[UNIFY] " << stage << std::endl;
  // }

  CheckSchedule(sch, "single_kernel.cc:161_end_" + name);

  return envelope;
}

Operation Schedule::unify(std::string name, std::string tag, Map<std::string, ObjectRef> attrs,
			  const Array<Tensor>& tensors, const Array<Dimension>& explicit_dimensions) {
  return CreateSingleKernel(*this, name, tag, attrs, tensors, tensors, true, explicit_dimensions,
                            {});
}

Operation Schedule::single_kernel(std::string name, std::string tag, Map<std::string, ObjectRef> attrs,
                               const Array<Tensor>& inputs_, const Array<Tensor>& outputs_,
                               bool include_inputs, const Array<IterVar>& thread_vars) {
  return CreateSingleKernel(*this, name, tag, attrs, inputs_, outputs_, include_inputs, {},
                            thread_vars);
}
}  // namespace te
}  // namespace tvm
