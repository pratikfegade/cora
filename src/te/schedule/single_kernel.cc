#include <tvm/te/operation.h>

#include "graph.h"
#include "schedule_utils.h"

#define COUT std::cout << "[SK] "
namespace tvm {
namespace te {

bool CheckSchedule(const Schedule& sch) {
  for (const auto& s : sch->stages) {
    std::cout << "[SK] " << s << " " << s->op << std::endl;
  }

  Array<Operation> roots;
  for (const auto& op : sch->outputs) {
    if (!roots.Contains(op)) {
      roots.push_back(op);
    }
  }
  auto fg = CreateFeedGraph(CreateReadGraph(roots));

  for (auto s : sch->stages) {
    for (int i = 0; i < s->op->num_outputs(); ++i) {
      Tensor t = s->op.output(i);
      CHECK(fg.count(t)) << t;
    }
  }
  return true;
}

Operation Schedule::single_kernel(std::string name, std::string tag,
                                  Map<std::string, ObjectRef> attrs, const Array<Tensor>& inputs,
                                  const Array<Tensor>& outputs, bool include_inputs,
                                  const Array<IterVar>& thread_vars) {
  CheckSchedule(*this);

  (*this)->InvalidateCache();

  /************** Create the envelope op **************/
  Operation envelope = SingleKernelEnvelopeOpNode::make(name, tag, attrs, outputs);

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
  std::cout << "[SK] RDF" << std::endl;
  ReplaceDataFlow((*this)->stages, &vmap, &rvmap);
  Stage envelope_stage = Stage(envelope);

  /************** Update stages **************/
  ArrayNode* stages = (*this)->stages.CopyOnWrite();
  size_t pos = 0;
  for (const auto& output : outputs) {
    pos = std::max(pos, FindNodeRef(stages, operator[](output->op)));
  }
  CHECK_LT(pos, stages->data.size());
  stages->data.insert(stages->data.begin() + pos + 1, envelope_stage);
  (*this)->stage_map.Set(envelope, envelope_stage);
  envelope_stage.env_threads(thread_vars);

  /************** Update schedule outputs **************/
  ArrayNode* sch_outputs = (*this)->outputs.CopyOnWrite();
  bool have_output = false;
  for (const auto& output : outputs) {
    Stage s = operator[](output->op);
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

  for (auto s : (*this)->stages) {
    CHECK(s->attach_type != kSingleKernelScope);
  }

  // /************** Create group **************/
  // // Create a group out of all the passed ops and attach that to the
  // // stage of he envelope op
  // Stage group = this->create_group(outputs, inputs, include_inputs);
  // group->attach_type = kSingleKernelScope;
  // group->attach_ivar = thread_vars[thread_vars.size() - 1];
  // group->attach_stage = envelope_stage;

  // return output_tensors;
  return envelope;
}
}  // namespace te
}  // namespace tvm
