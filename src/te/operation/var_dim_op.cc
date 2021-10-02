#include <tvm/te/operation.h>

namespace tvm {
namespace te {
using namespace tir;

IterVar BaseVarDimOpNode::GetIterVarFromDim(int val_idx, Dimension dim, bool only_loop_dims) const {
  CHECK_LT(val_idx, this->dim2var_maps.size()) << this->name;
  auto it = this->dim2var_maps[val_idx].find(dim.as<DimensionNode>());
  // if (it != this->dim2var_maps[val_idx].end()) {
  // for (auto it : this->dim2var_maps[val_idx]) {
  // std::cout << this->name << " " << it.first->name << std::endl;
  // }
  // }
  CHECK(it != this->dim2var_maps[val_idx].end())
      << "No such dimension " << dim->name << " in " << this->name;
  return it->second.iv;
}

Dimension BaseVarDimOpNode::GetDimensionFromVar(int val_idx, Var var) const {
  CHECK(var2dim_map.count(var.as<VarNode>())) << var << " " << name;
  return GetRef<Dimension>(var2dim_map.at(var.as<VarNode>()));
}

Array<DimInfo> BaseVarDimOpNode::GetAllDimensions() const {
  Array<DimInfo> ret;
  for (auto map : this->dim2var_maps) {
    for (auto it : map) {
      auto entry = it.second;
      ret.push_back(DimInfoNode::make(entry.dim, entry.iv));
    }
  }
  return ret;
}

}  // namespace te
}  // namespace tvm
