#ifndef TVM_TIR_UNINTERP_FUN_H_
#define TVM_TIR_UNINTERP_FUN_H_

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
  /*! \brief The range of the function */
  Range range;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("fname", &fname);
    v->Visit("paramters", &parameters);
    v->Visit("body", &body);
  }

  TVM_DLL static UninterpFun make(std::string fname, Range range,
                                  Array<tvm::te::Dimension> dimensions, Array<Var> parameters,
                                  PrimExpr body);

  TVM_DLL static UninterpFun from_constant(std::string fname, PrimExpr val);

  /*! \brief Get the name. */
  const std::string& func_name() const final { return fname; }

  int num_outputs() const;

  bool is_complex() const;

  void SetBody(PrimExpr expr);

  /*! \brief Get the arity. */
  size_t arity() const;

  int GetArgPos(Var var) const;

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

  static PrimExpr InlineUninterpFunCalls(PrimExpr e, bool only_simple = false);

  static Stmt InlineUninterpFunCalls(Stmt e, bool only_simple = false);

  static Range InlineUninterpFunCalls(Range r, bool only_simple = false);

  static Map<Dimension, PrimExpr> InvertCall(PrimExpr call, UninterpFun ufun);

  static ArgMappingAndEquality CheckEquality(UninterpFun f1, UninterpFun f2);

  static PrimExpr MakeCallTo(UninterpFun f, Array<PrimExpr> args, Array<Dimension> arg_dims);

  static PrimExpr RelaxComplexUninterpCalls(PrimExpr expr);
};

inline const UninterpFunNode* UninterpFun::operator->() const {
  return static_cast<const UninterpFunNode*>(data_.get());
}

////////////////////////////////////////////////////////////////////////////////////
// /*! \brief container class of iteration variable. */

// class Modes;

// /*!
//  * \brief Uninterpreted function node
//  */
// class ModesNode : public runtime::Object {
//  public:
//   /*! \brief named dimensions corresponding to the parameters */
//   Array<tvm::te::Dimension> dimensions;
//   /*! \brief functions representing the width of each dimension,
//    * potentially as a function of outer dimensions */
//   Array<UninterpFun> dim_widths;
//   /*! \brief optional functions representing the aggregate positions
//    * of each dimension, potentially as a function of outer
//    * dimensions */
//   Array<UninterpFun> dim_positions;

//   void VisitAttrs(AttrVisitor* v) {
//     v->Visit("dimensions", &dimensions);
//     v->Visit("dim_widths", &dim_widths);
//     v->Visit("dim_positions", &dim_positions);
//   }

//   TVM_DLL static Modes make(Array<tvm::te::Dimension> dimensions, Array<PrimExpr> dim_widths,
//                             Array<UninterpFun> dim_width_ufs, Array<UninterpFun>
//                             dim_position_ufs);

//   TVM_DLL static Modes make(std::string name, Array<PrimExpr> dim_widths);

//   /*! \brief Get dense overapproximated shape. */
//   const Array<PrimExpr> get_dense_shape() const;

//   /*! \brief Get number of dimensions. */
//   const size_t ndim() const { return dimensions.size(); };

//   const bool is_ragged() const;

//   const bool is_ragged(int i) const;

//   const PrimExpr ComputePosition(std::string name, Array<PrimExpr> coords) const;

//   const DataType get_dtype() const { return dim_widths[0]->body.dtype(); };

//   static constexpr const char* _type_key = "tir.Modes";
//   TVM_DECLARE_FINAL_OBJECT_INFO(ModesNode, Object);
// };

// /*!
//  * \brief Modes object to represent ragged tensor shapes and iteration
//  * spaces
//  */
// class Modes : public runtime::ObjectRef {
//  public:
//   Modes() {}
//   // construct from shared ptr.
//   explicit Modes(ObjectPtr<Object> n) : ObjectRef(n) {}
//   /*!
//    * \brief access the internal node container
//    * \return the pointer to the internal node container
//    */
//   inline const ModesNode* operator->() const;

//   /*! \brief specify container node */
//   using ContainerType = ModesNode;
// };

// inline const ModesNode* Modes::operator->() const {
//   return static_cast<const ModesNode*>(data_.get());
// }
////////////////////////////////////////////////////////////////////////////////////

}  // namespace tir
}  // namespace tvm

#endif
