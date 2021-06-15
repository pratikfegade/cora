#ifndef TVM_TIR_MODES_H_
#define TVM_TIR_MODES_H_

#include <tvm/arith/int_set.h>
#include <tvm/ir/expr.h>
#include <tvm/runtime/container.h>
#include <tvm/te/dimension.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>

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
  Array<UninterpFun> dim_widths;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("dimensions", &dimensions);
    v->Visit("dim_widths", &dim_widths);
  }

  TVM_DLL static Modes make(Array<tvm::te::Dimension> dimensions, Array<UninterpFun> dim_widths);

  TVM_DLL static Modes make(Array<tvm::te::Dimension> dimensions, Array<PrimExpr> dim_widths);

  TVM_DLL static Modes make(std::string name, Array<PrimExpr> dim_widths);

  /*! \brief Get dense overapproximated shape. */
  const Array<PrimExpr> get_dense_shape() const;

  /*! \brief Get number of dimensions. */
  const size_t ndim() const { return dimensions.size(); };

  /*! \brief Get number of dimensions. */
  const bool is_ragged() const;

  const DataType get_dtype() const { return dim_widths[0]->body.dtype(); };

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
