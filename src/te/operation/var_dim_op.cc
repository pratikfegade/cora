#include <tvm/te/operation.h>

namespace tvm {
namespace te {
using namespace tir;

DimVarEntry BaseVarDimOpNode::GetDimVarEntry(Dimension dim, bool only_loop_dims) const {
  auto it = this->dim2var_map.find(dim.as<DimensionNode>());
  CHECK(it != this->dim2var_map.end()) << "No such dimension " << dim->name;
  return it->second;
}

IterVar BaseVarDimOpNode::GetIterVarFromDim(Dimension dim, bool only_loop_dims) const {
  return GetDimVarEntry(dim, only_loop_dims).iv;
  // auto it = this->dim2var_map.find(dim.as<DimensionNode>());
  // CHECK(it != this->dim2var_map.end()) << "No such dimension " << dim->name;
  // return it->second.iv;
}

}
}
