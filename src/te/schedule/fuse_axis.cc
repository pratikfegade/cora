#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <tvm/te/schedule.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>

#include "../operation/op_util.h"
#include "graph.h"
#include "message_passing.h"
#include "ragged_utils.h"
#include "tensor_layout_utils.h"

namespace tvm {
namespace te {

class FuseRewriter : public ExprMutator {
 public:
  FuseRewriter(const Map<Operation, Operation>& rmap_, Dimension outer_, Dimension inner_,
               Dimension fused_, IterVar fused_iv_)
      : rmap(rmap_), outer(outer_), inner(inner_), fused(fused_), fused_iv(fused_iv_) {}

  PrimExpr VisitExpr_(const CallNode* op) override {
    std::cout << "[FA]  RRewriting " << GetRef<PrimExpr>(op) << " " << op->args.size() << " "
              << op->arg_dims.size() << std::endl;
    if (op->call_type == CallNode::Halide) {
      auto callee = Downcast<Operation>(op->func);
      if (!rmap.count(callee)) {
        std::cout << "[FA]   Couldn't find " << GetRef<PrimExpr>(op) << std::endl;
        return ExprMutator::VisitExpr_(op);
      }
      auto new_callee = rmap.at(callee);
      auto bvd_op = callee.as<BaseVarDimOpNode>();

      Array<PrimExpr> args;

      for (int i = 0; i < op->args.size(); ++i) {
        auto dim = bvd_op->GetBaseIndexDimension(0, i);
        if (dim == outer) {
          CHECK(op->args[i].as<VarNode>());
          // arg_dims.push_back(fused);
          args.push_back(fused_iv->var);
        } else if (dim == inner) {
          CHECK(op->args[i].as<VarNode>());
        } else {
          args.push_back(op->args[i]);
        }
      }

      std::cout << "[FA]   RReplaced" << std::endl;
      auto expr = CallNode::make(op->dtype, op->name, args, op->call_type, op->arg_dims, new_callee,
                                 op->value_index, op->custom_realize_bounds);
      std::cout << "[FA]     RReplacing " << expr << std::endl;
      return expr;
    } else {
      return ExprMutator::VisitExpr_(op);
    }
  }

  const Map<Operation, Operation>& rmap;
  Dimension outer;
  Dimension inner;
  Dimension fused;
  IterVar fused_iv;
};

Map<Operation, Operation> fuse_ragged_axis(Array<Tensor> input_tensors, Tensor output_tensor,
                                           Dimension outer, Dimension inner, Dimension fused,
                                           PrimExpr extent) {
  Array<Operation> graph_ops = GetSubGraph({output_tensor}, input_tensors, true);

  Map<Operation, Operation> rewritten;
  Map<Operation, Operation> ret;
  for (auto op : graph_ops) {
    std::cout << "[FA] Op " << op << std::endl;
    auto fused_op = op;
    auto bvd_op = op.as<BaseVarDimOpNode>();
    if (auto pop = op.as<PlaceholderOpNode>()) {
      Array<Dimension> dimensions = pop->self_index_dimensions;
      if (dimensions.Contains(outer) && dimensions.Contains(inner)) {
        IterVar ivo = pop->GetIterVarFromDim(0, outer);
        IterVar ivi = pop->GetIterVarFromDim(0, inner);

        IterVar ivf = IterVarNode::make(Range::make_by_min_extent(0, extent),
                                        Var("fused", DataType::Int(32)), kDataPar);
        auto layout = pop->layout;

        Array<Dimension> fused_dimensions;
        Array<UninterpFun> fused_uninterpfuns;
        Array<IterVar> fused_axis;
        Array<PrimExpr> fused_l_maxes;
        Array<UninterpFun> fused_l_funs;
        for (size_t i = 0; i < dimensions.size(); ++i) {
          auto dim = dimensions[i];
          auto iv = pop->GetIterVarFromDim(0, dim);
          if (dim == outer) {
            fused_dimensions.push_back(fused);
            fused_axis.push_back(ivf);
            fused_l_maxes.push_back(extent);
            fused_l_funs.push_back(
                UninterpFunNode::from_constant("f", extent, UninterpFunNode::kLFun));
            continue;
          } else if (dim == inner) {
            continue;
          }
          fused_dimensions.push_back(dim);
          fused_axis.push_back(iv);
          fused_l_maxes.push_back(layout->l_maxes[i]);
          fused_l_funs.push_back(layout->l_funs[i]);
          fused_uninterpfuns.push_back(NullValue<UninterpFun>());
        }

        auto fused_layout = ModesNode::make_storage_layout(
            fused_dimensions, fused_l_maxes, fused_l_funs, Map<Dimension, UninterpFun>());

        fused_op = PlaceholderOpNode::make(pop->name, fused_l_maxes, fused_layout, pop->dtype,
                                           fused_dimensions, fused_dimensions, fused_axis,
                                           fused_uninterpfuns);
      }
    } else if (auto cop = op.as<ComputeOpNode>()) {
      CHECK_EQ(op->num_outputs(), 1);
      Array<Dimension> dimensions = cop->root_index_dimensions;
      if (dimensions.Contains(outer) && dimensions.Contains(inner)) {
        IterVar ivo = cop->GetIterVarFromDim(0, outer);
        IterVar ivi = cop->GetIterVarFromDim(0, inner);

        IterVar ivf = IterVarNode::make(Range::make_by_min_extent(0, extent),
                                        Var("fused", DataType::Int(32)), kDataPar);

        auto layout = op->output_layout(0);
        auto llayout = op->loop_layout();
        Array<Dimension> fused_dimensions;
        Array<IterVar> fused_axis;
        Array<PrimExpr> fused_l_maxes;
        Array<UninterpFun> fused_l_funs;
        Array<PrimExpr> fused_ll_maxes;
        Array<UninterpFun> fused_ll_funs;
        Array<UninterpFun> fused_ll_fun_mins;
        for (size_t i = 0; i < dimensions.size(); ++i) {
          auto dim = dimensions[i];
          auto iv = cop->axis[i];
          if (dim == outer) {
            fused_dimensions.push_back(fused);
            fused_axis.push_back(ivf);
            if (layout.defined()) {
              fused_l_maxes.push_back(extent);
              fused_l_funs.push_back(
                  UninterpFunNode::from_constant("f", extent, UninterpFunNode::kLFun));
            }
            if (llayout.defined()) {
              fused_ll_maxes.push_back(extent);
              fused_ll_funs.push_back(
                  UninterpFunNode::from_constant("f", extent, UninterpFunNode::kLFun));
              fused_ll_fun_mins.push_back(
                  UninterpFunNode::from_constant("z", 0, UninterpFunNode::kLFun));
            }
            continue;
          } else if (dim == inner) {
            continue;
          }
          fused_dimensions.push_back(dim);
          fused_axis.push_back(iv);
          if (layout.defined()) {
            fused_l_maxes.push_back(layout->l_maxes[i]);
            fused_l_funs.push_back(layout->l_funs[i]);
          }
          if (llayout.defined()) {
            fused_ll_maxes.push_back(llayout->l_maxes[i]);
            fused_ll_funs.push_back(llayout->l_funs[i]);
            fused_ll_fun_mins.push_back(llayout->l_fun_mins[i]);
          }
        }

        FuseRewriter rewriter(rewritten, outer, inner, fused, ivf);
        Array<PrimExpr> fused_body;
        Array<PrimExpr> fused_pred;
        for (auto e : cop->body) {
          auto re = rewriter(e);
          std::cout << "[FA]  Body " << e << std::endl;
          std::cout << "[FA]   Replaced " << re << std::endl;
          fused_body.push_back(re);
        }
        for (auto e : cop->pred) {
          auto re = rewriter(e);
          std::cout << "[FA]  Pred " << e << " " << re << std::endl;
          fused_pred.push_back(re);
        }

        Array<Modes> fused_storage_layouts;
        if (layout.defined()) {
          fused_storage_layouts.push_back(ModesNode::make_storage_layout(
              fused_dimensions, fused_l_maxes, fused_l_funs, Map<Dimension, UninterpFun>()));
        }

        auto fused_loop_layout = NullValue<Modes>();
        if (llayout.defined()) {
          fused_loop_layout = ModesNode::make_loop_layout(fused_dimensions, fused_ll_maxes,
                                                          fused_ll_fun_mins, fused_ll_funs);
        }

        fused_op = ComputeOpNode::make(cop->name, cop->tag, cop->attrs, fused_axis,
                                       fused_dimensions, fused_l_maxes, fused_storage_layouts,
                                       fused_loop_layout, fused_body, fused_pred);
      }
    }
    if (!op.same_as(fused_op)) {
      rewritten.Set(op, fused_op);
    }
    ret.Set(op, fused_op);
  }
  return ret;
}

TVM_REGISTER_GLOBAL("te.FuseRaggedAxis")
    .set_body_typed([](Array<Tensor> input_tensors, Tensor output_tensor, Dimension outer,
                       Dimension inner, Dimension fused, PrimExpr extent) {
      std::cout << "[FA] Fusing Ragged Axis" << std::endl;
      return fuse_ragged_axis(input_tensors, output_tensor, outer, inner, fused, extent);
    });

}  // namespace te
}  // namespace tvm

#undef COUT
