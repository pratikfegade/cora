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
  /*! \brief Max extents for the l_funs. Stored for convenience */
  Array<PrimExpr> l_maxes;
  /*! \brief functions that together represent the width of each
   * dimension, potentially as a function of outer dimensions. Only loop layouts have defined
   * l_fun_mins, while l_funs are defined for both storage and loop layouts */
  Array<UninterpFun> l_funs;
  Array<UninterpFun> l_fun_mins;
  /*! \brief optional functions representing the aggregate positions
   * of each dimension, taking into consider all inner dimensions,
   * potentially as a function of outer dimensions */
  Array<UninterpFun> a_funs;
  /*! \brief Whether this modes object represents a loop nest */
  bool loop_layout;
  /*! \brief Map from a dimension to all dimensions that depend on it transitively wrt l_funs */
  mutable Map<Dimension, Array<Dimension>> transitive_dependent_dims;
  /*! \brief Map from a dimension to all dimensions that immediately depend on it wrt l_funs */
  mutable Map<Dimension, Array<Dimension>> immediate_dependent_dims;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("dimensions", &dimensions);
    v->Visit("l_funs", &l_funs);
    v->Visit("l_fun_mins", &l_fun_mins);
    v->Visit("a_funs", &a_funs);
    v->Visit("transitive_dependent_dims", &transitive_dependent_dims);
    v->Visit("immediate_dependent_dims", &immediate_dependent_dims);
    v->Visit("loop_layout", &loop_layout);
  }

  TVM_DLL static Modes make(Array<tvm::te::Dimension> dimensions, Array<PrimExpr> l_maxes,
                            Array<UninterpFun> l_fun_mins_, Array<UninterpFun> l_funs,
                            Array<UninterpFun> a_funs, bool is_loop_layout);

  TVM_DLL static Modes make(Array<tvm::te::Dimension> dimensions, Array<PrimExpr> l_maxes,
                            Array<UninterpFun> l_fun_mins_, Array<UninterpFun> l_funs,
                            Map<Dimension, UninterpFun> user_a_funs, bool is_loop_layout);

  TVM_DLL static Modes make_loop_layout(Array<tvm::te::Dimension> dimensions,
                                        Array<PrimExpr> l_maxes, Array<UninterpFun> l_fun_mins,
                                        Array<UninterpFun> l_funs);

  TVM_DLL static Modes make_storage_layout(Array<tvm::te::Dimension> dimensions,
                                           Array<PrimExpr> l_maxes, Array<UninterpFun> l_funs,
                                           Array<UninterpFun> a_funs);

  TVM_DLL static Modes make_storage_layout(Array<tvm::te::Dimension> dimensions,
                                           Array<PrimExpr> l_maxes, Array<UninterpFun> l_funs,
                                           Map<Dimension, UninterpFun> user_a_funs);

  TVM_DLL static Modes make(std::string name, Array<PrimExpr> dense_shape, bool is_loop_layout);

  /*! \brief Get dense overapproximated shape. */
  const Array<PrimExpr> get_dense_shape() const;

  /*! \brief Get number of dimensions. */
  const size_t ndim() const { return dimensions.size(); };

  const bool is_ragged() const;

  const bool is_ragged(int i) const;

  const std::string str() const;

  const bool has_dependent_dims(int idx) const;

  const Array<Dimension> get_transitive_dependent_dims(int idx) const;

  const Array<Dimension> get_immediate_dependent_dims(int idx) const;

  const PrimExpr ComputePosition(std::string name, Array<PrimExpr> coords) const;

  const PrimExpr ComputePosition(std::string name, Array<PrimExpr> coords,
                                 Array<Dimension> relevant_dims) const;

  const PrimExpr GetAllocationSize() const;

  const DataType get_dtype() const { return DataType::Int(32); };

  static constexpr const char* _type_key = "tir.Modes";
  TVM_DECLARE_FINAL_OBJECT_INFO(ModesNode, Object);

 private:
  const void setup_transitive_dependences() const;
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
