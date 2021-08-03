#include <tvm/runtime/registry.h>
#include <tvm/te/dimension.h>
#include <tvm/te/dimension_relations.h>

namespace tvm {
namespace te {
TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<DimensionNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const DimensionNode*>(node.get());
      p->stream << "Dim(" << op->name << ", " << op << ")";
    });

Dimension DimensionNode::make(std::string name, DimensionNode::DimensionType type) {
  ObjectPtr<DimensionNode> n = make_object<DimensionNode>();
  n->name = name;
  n->type = type;
  return Dimension(n);
}

bool DimensionNode::isFunDim() const { return this->type == DimensionNode::DimensionType::kFunDim; }

bool DimensionNode::isRangeDim() const {
  return this->type == DimensionNode::DimensionType::kRangeDim;
}

bool DimensionNode::isScanDim() const {
  return this->type == DimensionNode::DimensionType::kScanDim;
}

bool DimensionNode::isLoopDim() const {
  return this->type == DimensionNode::DimensionType::kScanDim ||
         this->type == DimensionNode::DimensionType::kRangeDim;
}

TVM_REGISTER_NODE_TYPE(DimensionNode);
TVM_REGISTER_GLOBAL("te.RangeDimension").set_body_typed([](std::string name) {
  return DimensionNode::make(name, DimensionNode::DimensionType::kRangeDim);
});

TVM_REGISTER_GLOBAL("te.ScanDimension").set_body_typed([](std::string name) {
  return DimensionNode::make(name, DimensionNode::DimensionType::kScanDim);
});

DimensionRelation DimensionSplitNode::make(Dimension parent, Dimension outer, Dimension inner,
                                           PrimExpr factor, PrimExpr nparts) {
  ObjectPtr<DimensionSplitNode> n = make_object<DimensionSplitNode>();
  n->parent = parent;
  n->outer = outer;
  n->inner = inner;
  n->factor = factor;
  n->nparts = nparts;
  return DimensionRelation(n);
}

DimensionRelation DimensionFuseNode::make(Dimension outer, Dimension inner, Dimension fused,
                                          int factor) {
  ObjectPtr<DimensionFuseNode> n = make_object<DimensionFuseNode>();
  n->outer = outer;
  n->inner = inner;
  n->fused = fused;
  n->factor = factor;
  return DimensionRelation(n);
}

DimensionRelation RaggedDimensionFuseNode::make(Dimension outer, Dimension inner, Dimension fused,
                                                tir::UninterpFun fused_to_outer_uf,
                                                tir::UninterpFun fused_to_inner_uf,
                                                tir::UninterpFun outer_inner_to_fused_uf) {
  ObjectPtr<RaggedDimensionFuseNode> n = make_object<RaggedDimensionFuseNode>();
  n->outer = outer;
  n->inner = inner;
  n->fused = fused;
  n->fused_to_outer_uf = fused_to_outer_uf;
  n->fused_to_inner_uf = fused_to_inner_uf;
  n->outer_inner_to_fused_uf = outer_inner_to_fused_uf;
  return DimensionRelation(n);
}

DimensionRelationGraph DimensionRelationGraphNode::make(Array<Dimension> root_dimensions) {
  ObjectPtr<DimensionRelationGraphNode> n = make_object<DimensionRelationGraphNode>();
  n->leaf_dimensions = root_dimensions;
  n->root_dimensions = root_dimensions;
  return DimensionRelationGraph(n);
}

std::string op2str(const DimKey::OpType& op) {
  switch (op) {
    case DimKey::kFuse:
      return "fuse";
    case DimKey::kSplitOuter:
      return "split_outer";
    case DimKey::kSplitInner:
      return "split_inner";
    case DimKey::kRebase:
      return "rebase";
    default:
      return "What?";
  }
}

std::unordered_map<DimKey, const DimensionNode*, DimKeyHasher, DimKeyEquality>
    Dimension::op_dim_map;

Dimension Dimension::get_or_create_dimension(const DimKey& key) {
  auto it = op_dim_map.find(key);
  if (it != op_dim_map.end()) {
    return GetRef<Dimension>(it->second);
  } else {
    auto name = op2str(key.op) + std::to_string(op_dim_map.size());
    auto dim = DimensionNode::make(name, DimensionNode::kRangeDim);
    op_dim_map[key] = dim.operator->();
    return dim;
  }
}
}  // namespace te
}  // namespace tvm
