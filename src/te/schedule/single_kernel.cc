#include <tvm/te/operation.h>
#include "schedule_utils.h"

#define COUT std::cout << "[SK] "
namespace tvm {
namespace te {

Array<Tensor> Schedule::single_kernel(std::string name,
				      std::string tag,
				      Map<std::string, ObjectRef> attrs,
				      const Array<Tensor>& inputs,
				      const Array<Tensor>& outputs,
				      bool include_inputs,
				      const Array<IterVar>& thread_vars) {
  (*this)->InvalidateCache();
  // Create the envelope op
  Operation envelope = SingleKernelEnvelopeOpNode::make(name, tag, attrs, outputs);

  std::unordered_map<Tensor, Tensor> vmap;
  std::unordered_map<Tensor, Tensor> rvmap;
  Array<Tensor> output_tensors;
  for (size_t i = 0; i < inputs.size(); ++i) {
    Tensor output = envelope.output(i);
    vmap[inputs[i]] = output;
    rvmap[output] = inputs[i];
    output_tensors.push_back(output);
  }
  ReplaceDataFlow((*this)->stages, &vmap, &rvmap);
  Stage envelope_stage = Stage(envelope);

  ArrayNode* stages = (*this)->stages.CopyOnWrite();
  size_t pos = 0;
  for (const auto& input: inputs) {
    pos = std::max(pos, FindNodeRef(stages, operator[](input->op)));
  }
  CHECK_LT(pos, stages->data.size());
  stages->data.insert(stages->data.begin() + pos + 1,
		      envelope_stage);
  (*this)->stage_map.Set(envelope, envelope_stage);
  envelope_stage.env_threads(thread_vars);

  // Create a group out of all the passed ops and attach that to the
  // stage of he envelope op
  Stage group = this->create_group(outputs, inputs, include_inputs);
  group->attach_type = kScope;
  group->attach_ivar = thread_vars[thread_vars.size() - 1];
  group->attach_stage = envelope_stage;

  return output_tensors;
}
}
}
