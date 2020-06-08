#include <tvm/te/operation.h>

namespace tvm {
namespace te {
using namespace tir;

DimVarEntry BaseVarDimOpNode::GetDimVarEntry(int val_idx, Dimension dim,
                                             bool only_loop_dims) const {
  CHECK_LT(val_idx, this->dim2var_maps.size()) << this->name;
  auto it = this->dim2var_maps[val_idx].find(dim.as<DimensionNode>());
  CHECK(it != this->dim2var_maps[val_idx].end()) << "No such dimension " << dim->name;
  return it->second;
}

IterVar BaseVarDimOpNode::GetIterVarFromDim(int val_idx, Dimension dim, bool only_loop_dims) const {
  return GetDimVarEntry(val_idx, dim, only_loop_dims).iv;
}

DimVarEntry BaseVarDimOpNode::GetDimVarEntry(int val_idx, Var var) const {
  return GetDimVarEntry(val_idx, GetRef<Dimension>(var2dim_map.at(var.as<VarNode>())));
}
}  // namespace te
}  // namespace tvm
