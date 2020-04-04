#ifndef TVM_TE_SCHEDULE_UTILS_H_
#define TVM_TE_SCHEDULE_UTILS_H_

#include <tvm/tir/expr.h>
#include <tvm/te/tensor.h>

namespace tvm {
namespace te {
// find first occurance location in leaf
template<typename T>
size_t FindNodeRef(ArrayNode* array_node, const T& v);

void ReplaceDataFlow(const Array<Stage>& stages,
                     std::unordered_map<Tensor, Tensor>* vmap,
                     std::unordered_map<Tensor, Tensor>* rvmap);

}
}

#endif
