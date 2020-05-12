#ifndef TVM_TE_SCHEDULE_UTILS_H_
#define TVM_TE_SCHEDULE_UTILS_H_

#include <tvm/te/tensor.h>
#include <tvm/tir/expr.h>

namespace tvm {
namespace te {
Array<Tensor> RemapTensor(ScheduleNode* self, const Array<Tensor>& arr);

bool CheckSchedule(const Schedule& sch, const std::string& err = "bounds ");

// find first occurance location in leaf
void ReplaceDataFlow(const Array<Stage>& stages, std::unordered_map<Tensor, Tensor>* vmap,
                     std::unordered_map<Tensor, Tensor>* rvmap);

// find first occurance location in leaf
template <typename T>
size_t FindNodeRef(ArrayNode* array_node, const T& v) {
  const Object* n = v.get();
  for (size_t i = 0; i < array_node->data.size(); ++i) {
    if (array_node->data[i].get() == n) return i;
  }
  return array_node->data.size();
}

}  // namespace te
}  // namespace tvm

#endif
