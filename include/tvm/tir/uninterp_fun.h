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

class RaggedFusionInfo;

class RaggedFusionInfoNode : public runtime::Object {
 public:
  /*! \brief The outer domain */
  IterVar outer;
  /*! \brief The inner domain */
  IterVar inner;
  /*! \brief The target domain */
  IterVar fused;
  /*! \brief Parent to outer relation uf */
  FunctionRef fused_to_outer_uf;
  /*! \brief Parent to inner relation uf */
  FunctionRef fused_to_inner_uf;
  /*! \brief inner and outer to parent relation uf */
  FunctionRef outer_inner_to_fused_uf;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("outer", &outer);
    v->Visit("inner", &inner);
    v->Visit("fused", &fused);
    v->Visit("outer", &outer);
    v->Visit("fused_to_outer_uf", &fused_to_outer_uf);
    v->Visit("fused_to_inner_uf", &fused_to_inner_uf);
    v->Visit("outer_inner_to_fused_uf", &outer_inner_to_fused_uf);
  }

  TVM_DLL static RaggedFusionInfo make(IterVar outer, IterVar inner, IterVar fused,
                                       FunctionRef fused_to_outer_uf, FunctionRef fused_to_inner_uf,
                                       FunctionRef outer_inner_to_fused_uf);

  static constexpr const char* _type_key = "te.RaggedFusionInfo";
  TVM_DECLARE_FINAL_OBJECT_INFO(RaggedFusionInfoNode, Object);
};

class RaggedFusionInfo : public runtime::ObjectRef {
 public:
  static RaggedFusionInfo NoRaggedFusionInfo;

  RaggedFusionInfo() {}
  // construct from shared ptr.
  explicit RaggedFusionInfo(runtime::ObjectPtr<runtime::Object> n) : ObjectRef(n) {}
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const RaggedFusionInfoNode* operator->() const;

  /*! \brief specify container node */
  using ContainerType = RaggedFusionInfoNode;
};

inline const RaggedFusionInfoNode* RaggedFusionInfo::operator->() const {
  return static_cast<const RaggedFusionInfoNode*>(data_.get());
}

/*!
 * \brief Uninterpreted function node
 */
class UninterpFunNode : public FunctionBaseNode {
 public:
  enum UninterpFunType : int {
    kAFun = 0,
    kLFun = 1,
    kFOFun = 2,
    kFIFun = 3,
    kOIFFun = 4,
    kUnspecifiedFun = 5
  };

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
  /*! \brief The kind of uinterpreted function */
  UninterpFunType type;
  /*! \brief Used for FO and FI funs to maintain pointers to fields of
      the RaggedFusedSplitNode */
  RaggedFusionInfo fusion_info;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("fname", &fname);
    v->Visit("paramters", &parameters);
    v->Visit("dimensions", &dimensions);
    v->Visit("body", &body);
    v->Visit("range", &range);
    v->Visit("type", &type);
    v->Visit("fusion_info", &fusion_info);
  }

  TVM_DLL static UninterpFun make(std::string fname, Range range,
                                  Array<tvm::te::Dimension> dimensions, Array<Var> parameters,
                                  PrimExpr body, UninterpFunType type = kUnspecifiedFun);

  TVM_DLL static UninterpFun from_constant(std::string fname, PrimExpr val,
                                           UninterpFunType type = kUnspecifiedFun);

  /*! \brief Get the name. */
  const std::string& func_name() const final { return fname; }

  int num_outputs() const;

  bool is_complex() const;

  bool is_constant() const;

  void SetBody(PrimExpr expr);

  void SetRange(Range r);

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

  const PrimExpr MakeCallTo(Array<Var> arg_vars, Array<Dimension> arg_dims,
                            DataType dtype = DataType::Handle()) const {
    Array<PrimExpr> args;
    for (auto param : arg_vars) {
      args.push_back(param);
    }
    return MakeCallTo(args, arg_dims, dtype);
  };

  const PrimExpr MakeCallTo(Array<PrimExpr> args, Array<Dimension> arg_dims,
                            DataType dtype = DataType::Handle()) const;

  static PrimExpr InlineUninterpFunCalls(PrimExpr e, bool only_simple = false);

  static Stmt InlineUninterpFunCalls(Stmt e, bool only_simple = false);

  static Range InlineUninterpFunCalls(Range r, bool only_simple = false);

  static Map<Dimension, PrimExpr> InvertCall(PrimExpr call, UninterpFun ufun);

  static ArgMappingAndEquality CheckEquality(UninterpFun f1, UninterpFun f2);

  static PrimExpr RelaxUninterpCallsMaxInclusive(PrimExpr expr, bool complex_only = true);
};

inline const UninterpFunNode* UninterpFun::operator->() const {
  return static_cast<const UninterpFunNode*>(data_.get());
}

}  // namespace tir
}  // namespace tvm

#endif
