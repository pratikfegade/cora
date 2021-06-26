#ifndef TVM_TIR_MODES_H_
#define TVM_TIR_MODES_H_

/* #include <tvm/arith/int_set.h> */
#include <tvm/ir/expr.h>
#include <tvm/runtime/container.h>
#include <tvm/te/dimension.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/uninterp_fun.h>

#include <vector>

namespace tvm {
namespace te {
/*! \brief container class of iteration variable. */
class Dimension;
}  // namespace te

namespace tir {
/*! \brief container class of iteration variable. */

class Modes;

/*!
 * \brief Uninterpreted function node
 */
class ModesNode : public runtime::Object {
 public:
  /*! \brief named dimensions corresponding to the parameters */
  Array<tvm::te::Dimension> dimensions;
  /*! \brief functions representing the width of each dimension,
   * potentially as a function of outer dimensions */
  Array<UninterpFun> l_funs;
  /*! \brief optional functions representing the aggregate positions
   * of each dimension, taking into consider all inner dimensions,
   * potentially as a function of outer dimensions */
  Array<UninterpFun> a_funs;
  /*! \brief Map from a dimension to all dimensions that depend on it transitively wrt l_funs */
  mutable Map<Dimension, Array<Dimension>> transitive_dependent_dims;
  /*! \brief Whether this modes object represents a loop nest */
  bool loop_layout;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("dimensions", &dimensions);
    v->Visit("l_funs", &l_funs);
    v->Visit("a_funs", &a_funs);
    v->Visit("transitive_dependent_dims", &transitive_dependent_dims);
    v->Visit("loop_layout", &loop_layout);
  }

  TVM_DLL static Modes make(Array<tvm::te::Dimension> dimensions, Array<PrimExpr> l_maxes,
                            Array<UninterpFun> l_funs, Array<UninterpFun> a_funs,
                            bool loop_layout = false);

  TVM_DLL static Modes make(std::string name, Array<PrimExpr> dim_widths);

  /*! \brief Get dense overapproximated shape. */
  const Array<PrimExpr> get_dense_shape() const;

  /*! \brief Get number of dimensions. */
  const size_t ndim() const { return dimensions.size(); };

  const bool is_ragged() const;

  const bool is_ragged(int i) const;

  const Array<Dimension> get_dependent_dimensions(Dimension dim) const;

  const std::string str() const;

  const void SetupTransitiveDependences() const;

  const PrimExpr ComputePosition(std::string name, Array<PrimExpr> coords) const;

  const PrimExpr ComputePositionOld(std::string name, Array<PrimExpr> coords) const;

  const PrimExpr ComputePosition(std::string name, Array<PrimExpr> coords,
                                 Array<Dimension> relevant_dims) const;

  const PrimExpr GetAllocationSize() const;

  const DataType get_dtype() const { return DataType::Int(32); };

  static constexpr const char* _type_key = "tir.Modes";
  TVM_DECLARE_FINAL_OBJECT_INFO(ModesNode, Object);
};

/*!
 * \brief Modes object to represent ragged tensor shapes and iteration
 * spaces
 */
class Modes : public runtime::ObjectRef {
 public:
  Modes() {}
  // construct from shared ptr.
  explicit Modes(ObjectPtr<Object> n) : ObjectRef(n) {}
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const ModesNode* operator->() const;

  /*! \brief specify container node */
  using ContainerType = ModesNode;
};

inline const ModesNode* Modes::operator->() const {
  return static_cast<const ModesNode*>(data_.get());
}
}  // namespace tir
}  // namespace tvm

#endif
