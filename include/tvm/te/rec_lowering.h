#include <tvm/te/schedule.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/expr.h>

namespace tvm {
namespace te {
class ILAOps;

class ILAOpsNode : public runtime::Object {
 public:
  Array<Tensor> ds_tensors;
  Array<Operation> outputs;
  Map<Tensor, Array<Tensor>> ra_ila_mapping;
  Map<std::string, Dimension> ds_dimensions;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("ds_tensors", &ds_tensors);
    v->Visit("outputs", &outputs);
    v->Visit("ra_ila_mapping", &ra_ila_mapping);
    v->Visit("ds_dimensions", &ds_dimensions);
  }

  TVM_DLL static ILAOps make(Array<Tensor> ds_tensors, Array<Operation> outputs,
                             Map<Tensor, Array<Tensor>> ra_ila_mapping,
                             Map<std::string, Dimension> ds_dimensions);

  static constexpr const char* _type_key = "ILAOps";
  TVM_DECLARE_FINAL_OBJECT_INFO(ILAOpsNode, Object);
};

class ILAOps : public runtime::ObjectRef {
 public:
  ILAOps() {}
  // construct from shared ptr.
  explicit ILAOps(runtime::ObjectPtr<runtime::Object> n) : ObjectRef(n) {}
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const ILAOpsNode* operator->() const;

  /*! \brief specify container node */
  using ContainerType = ILAOpsNode;
};

inline const ILAOpsNode* ILAOps::operator->() const {
  return static_cast<const ILAOpsNode*>(data_.get());
}

enum ScanRange : int { kAll = 0, kLeavesOnly = 1, kRootsOnly = 2, kNoLeaves = 3 };

ILAOps LowerDynamicBatching(Array<Operation> outputs, Var num_nodes, Var num_batches,
                            Var max_batch_len, Var max_child_num, Var max_int_idx,
                            bool leaf_specialization, bool is_list = false,
                            bool homogenous_batch = false, int batch_size = -1, int length = -1,
			    ScanRange scan_range = kAll);

ILAOps LowerStaticBatching(Array<Operation> outputs, Var num_nodes, PrimExpr num_trees,
                           Var max_tree_len, Var max_child_num);
}  // namespace te
}  // namespace tvm
