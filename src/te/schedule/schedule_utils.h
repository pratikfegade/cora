#ifndef TVM_TE_SCHEDULE_UTILS_H_
#define TVM_TE_SCHEDULE_UTILS_H_

#include <tvm/te/schedule.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/expr.h>

#include "graph.h"

namespace tvm {
namespace te {
bool isCudaThread(const IterVar& iv);

bool isCudaThread(const std::string& name);

bool isCPUEnvThread(const IterVar& iv);

bool isCPUEnvThread(const std::string& name);

bool equalCudaThreads(const IterVar& iv1, const IterVar& iv2);

Array<Tensor> RemapTensor(ScheduleNode* self, const Array<Tensor>& arr);

bool CheckSchedule(Schedule& sch, const std::string& caller = "None", bool print = false);

void ReplaceDataFlow(const Array<Stage>& stages, Map<FunctionRef, CacheInfo> cacheMappings,
                     std::unordered_map<Tensor, Tensor>* vmap,
                     std::unordered_map<Tensor, Tensor>* rvmap,
                     std::unordered_set<const OperationNode*> to_skip = {});
FeedGraph GetFeedGraph(Schedule& sch, bool includeUnemittedInputs, bool print = false);

ReadGraph GetReadGraph(Schedule& sch, bool includeUnemittedInputs, bool print = false);

// find first occurance location in leaf
template <typename T>
size_t FindNodeRef(ArrayNode* array_node, const T& v) {
  const Object* n = v.get();
  for (size_t i = 0; i < array_node->data.size(); ++i) {
    if (array_node->data[i].get() == n) return i;
  }
  return array_node->data.size();
}

template <typename T>
std::string array_to_str(Array<T> array) {
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < array.size(); ++i) {
    ss << array[i];
    if (i != array.size() - 1) ss << ", ";
  }
  ss << "]";
  return ss.str();
}

}  // namespace te
}  // namespace tvm

#endif
