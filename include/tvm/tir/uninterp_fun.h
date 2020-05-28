#ifndef TVM_TIR_UNINTERP_FUN_H_
#define TVM_TIR_UNINTERP_FUN_H_

#include <tvm/ir/expr.h>
#include <tvm/runtime/container.h>
#include <tvm/te/dimension.h>
#include <tvm/tir/expr.h>

#include <vector>

namespace tvm {
namespace te {
/*! \brief container class of iteration variable. */
class Dimension;
}  // namespace te

namespace tir {
/*! \brief container class of iteration variable. */

class UninterpFun;

struct ArgMappingAndEquality {
  bool equals;
  Map<Var, Var> mapping;
};

/*!
 * \brief Uninterpreted function node
 */
class UninterpFunNode : public FunctionBaseNode {
 public:
  /*!
   * \brief the name of the function
   */
  std::string fname;
  /*! \brief the parameters */
  Array<Var> parameters;
  /*! \brief named dimensions corresponding to the parameteres */
  Array<tvm::te::Dimension> dimensions;
  /*! \brief The body if the function */
  PrimExpr body;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("fname", &fname);
    v->Visit("paramters", &parameters);
    v->Visit("body", &body);
  }

  TVM_DLL static UninterpFun make(std::string fname, Range range, Array<Var> parameters,
                                  PrimExpr body);

  TVM_DLL static UninterpFun make(std::string fname, Range range,
                                  Array<tvm::te::Dimension> dimensions, Array<Var> parameters,
                                  PrimExpr body);

  /*! \brief Get the name. */
  const std::string& func_name() const final { return fname; }

  int num_outputs() const;

  bool is_complex() const;

  Range range;

  void SetBody(PrimExpr expr);

  /*! \brief Get the arity. */
  size_t arity() const;

  int GetArgPos(Var var) const;

  UninterpFun FunWithNewParams(Array<PrimExpr> param_exprs, Array<Var> new_params) const;

  const PrimExpr substitute(Array<PrimExpr> arguments, Array<tvm::te::Dimension> dimensions) const;

  static constexpr const char* _type_key = "tir.UninterpFun";
  TVM_DECLARE_FINAL_OBJECT_INFO(UninterpFunNode, Object);
};

/*!
 * \brief Uninterpreted function
 */
class UninterpFun : public FunctionRef {
 public:
  UninterpFun() {}
  // construct from shared ptr.
  explicit UninterpFun(ObjectPtr<Object> n) : FunctionRef(n) {}
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const UninterpFunNode* operator->() const;

  /*! \brief specify container node */
  using ContainerType = UninterpFunNode;

  static PrimExpr InlineUninterpFunCalls(PrimExpr e);

  static Range InlineUninterpFunCalls(Range r);

  static Map<Dimension, PrimExpr> InvertCall(PrimExpr call, UninterpFun ufun);

  static ArgMappingAndEquality CheckEquality(UninterpFun f1, UninterpFun f2);

  static PrimExpr MakeCallTo(UninterpFun f, Array<PrimExpr> args, Array<Dimension> arg_dims);

  static PrimExpr RelaxComplexUninterpCalls(PrimExpr expr);
};

inline const UninterpFunNode* UninterpFun::operator->() const {
  return static_cast<const UninterpFunNode*>(data_.get());
}
}  // namespace tir
}  // namespace tvm

#endif
