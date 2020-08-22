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
TVM_REGISTER_GLOBAL("te.FunDimension").set_body_typed([](std::string name) {
  return DimensionNode::make(name, DimensionNode::DimensionType::kFunDim);
});

TVM_REGISTER_GLOBAL("te.RangeDimension").set_body_typed([](std::string name) {
  return DimensionNode::make(name, DimensionNode::DimensionType::kRangeDim);
});

TVM_REGISTER_GLOBAL("te.ScanDimension").set_body_typed([](std::string name) {
  return DimensionNode::make(name, DimensionNode::DimensionType::kScanDim);
});

Dimension Dimension::NoDimension =
    DimensionNode::make("NoDim", DimensionNode::DimensionType::kRangeDim);

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

DimensionRelation DimensionFuseNode::make(Dimension outer, Dimension inner, Dimension fused) {
  ObjectPtr<DimensionFuseNode> n = make_object<DimensionFuseNode>();
  n->outer = outer;
  n->inner = inner;
  n->fused = fused;
  return DimensionRelation(n);
}

DimensionRelation DimensionChangeNode::make(Array<Dimension> old_dims, Array<Dimension> new_dims) {
  ObjectPtr<DimensionChangeNode> n = make_object<DimensionChangeNode>();
  n->old_dims = old_dims;
  n->new_dims = new_dims;
  return DimensionRelation(n);
}

DimensionRelationGraph DimensionRelationGraphNode::make(Array<Dimension> root_dimensions) {
  ObjectPtr<DimensionRelationGraphNode> n = make_object<DimensionRelationGraphNode>();
  n->leaf_dimensions = root_dimensions;
  n->root_dimensions = root_dimensions;
  return DimensionRelationGraph(n);
}
}  // namespace te
}  // namespace tvm
