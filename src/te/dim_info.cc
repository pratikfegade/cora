#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>

namespace tvm {
namespace te {
TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<DimInfoNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const DimInfoNode*>(node.get());
      p->stream << "Dim(" << op->dim << ", " << op << ")";
    });

DimInfo DimInfoNode::make(Dimension dim, IterVar iv) {
  ObjectPtr<DimInfoNode> n = make_object<DimInfoNode>();
  n->dim = dim;
  n->iv = iv;
  return DimInfo(n);
}

}  // namespace te
}  // namespace tvm
