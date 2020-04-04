#include <tvm/runtime/registry.h>
#include <tvm/te/dimension.h>

namespace tvm {
  namespace te {
    Dimension DimensionNode::make(std::string name, DimensionNode::DimensionType type) {
      ObjectPtr<DimensionNode> n = make_object<DimensionNode>();
      n->name = name;
      n->type = type;
      return Dimension(n);
    }

    TVM_REGISTER_NODE_TYPE(DimensionNode);
    TVM_REGISTER_GLOBAL("te.FunDimension")
    .set_body_typed([](std::string name) {
		      return DimensionNode::make(name, DimensionNode::DimensionType::kFunDim);
		    });

    TVM_REGISTER_GLOBAL("te.RangeDimension")
    .set_body_typed([](std::string name) {
		      return DimensionNode::make(name, DimensionNode::DimensionType::kRangeDim);
		    });

    Dimension Dimension::NoDimension = DimensionNode::make("NoDim", DimensionNode::DimensionType::kRangeDim);
  }
}
