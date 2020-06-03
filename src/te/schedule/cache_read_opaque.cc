#include <tvm/ir/attrs.h>
#include <tvm/te/operation.h>
#include <tvm/te/schedule.h>
#include <tvm/tir/expr_equality.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/tir/stmt_functor.h>

#include <unordered_set>

#include "../../arith/compute_expr.h"
#include "../../tir/ir/var_replacer.h"
#include "../../tir/pass/ir_util.h"
#include "graph.h"
#include "message_passing.h"
#include "schedule_utils.h"
#include "tensor_layout_utils.h"

#define COUT std::cout << "[CRO] "
namespace tvm {
namespace te {
PrimExpr CacheBodyBuilder(Tensor tensor, Array<Dimension>& original_index_dimensions,
                          const PatternsVec& patterns_vec, Array<IterVar>& index_variables,
                          Array<IterVar>& loop_variables, Array<Dimension>& index_dimensions,
                          Array<Dimension>& loop_dimensions) {
  const Var variant_loop_var = loop_variables[loop_variables.size() - 1]->var;
  PrimExpr body = PrimExpr(0);
  for (size_t i = 0; i < patterns_vec.size(); ++i) {
    AccessPattern* pattern = patterns_vec[i];
    PrimExpr expr;
    Array<PrimExpr> args;

    for (auto it : pattern->idx_dim_args) {
      std::cout << it.first << " " << it.second << std::endl;
    }

    for (size_t i = 0; i < original_index_dimensions.size(); ++i) {
      if (original_index_dimensions[i]->type == DimensionNode::kFunDim) {
        Dimension arg_dim = pattern->idx_dim_args.at(original_index_dimensions[i]);
        IterVar iv;
        if (index_dimensions.Contains(arg_dim)) {
          iv = index_variables[index_dimensions.GetIdx(arg_dim)];
        } else {
          auto reader_iv = pattern->reader_op->GetIterVarFromDim(pattern->reader_val_idx, arg_dim);
          index_dimensions.push_back(arg_dim);
          iv = IterVarNode::make(reader_iv->dom, reader_iv->var.copy_with_suffix(""),
                                 reader_iv->iter_type, reader_iv->thread_tag);
          index_variables.push_back(iv);
        }
        args.push_back(iv->var);
        // std::cout << "Arg  " << iv << std::endl;
      } else {
        IterVar iv = GetIterVarFromDim(original_index_dimensions[i], index_variables,
                                       loop_variables, index_dimensions, loop_dimensions);
        // std::cout << "Arg2  " << iv << std::endl;
        args.push_back(iv);
      }
    }

    expr = CallNode::make(tensor->op->output_dtype(tensor->value_index), tensor->op->name, args,
                          CallNode::Halide, tensor->op, 0);
    body = if_then_else(variant_loop_var == static_cast<int>(i), expr, body);
  }
  return body;
}

Tensor Schedule::cache_read_opaque(const Tensor& tensor, const std::string& scope,
                                   const Array<Operation>& readers, const std::string& suffix) {
  /************* Collect patterns *************/
  const ComputeOpNode* compute_op = tensor->op.as<ComputeOpNode>();
  const PlaceholderOpNode* placeholder_op = tensor->op.as<PlaceholderOpNode>();
  Array<IterVar> original_loop_axis;
  Array<Dimension> original_loop_dimensions;
  Array<UninterpFun> original_index_expressions;
  Array<Dimension> original_index_dimensions;
  Array<Dimension> original_root_index_dimensions;
  if (compute_op) {
    original_loop_axis = compute_op->axis;
    original_loop_dimensions = compute_op->loop_dimensions;
    original_index_expressions = compute_op->index_expressions;
    original_index_dimensions = compute_op->index_dimensions;
    original_root_index_dimensions = compute_op->root_index_dimensions;
  } else {
    original_loop_axis = placeholder_op->axis;
    original_loop_dimensions = placeholder_op->loop_dimensions;
    original_index_expressions = placeholder_op->index_expressions;
    original_index_dimensions = placeholder_op->index_dimensions;
    original_root_index_dimensions = placeholder_op->self_index_dimensions;
  }

  AccessPatternCollector collector(tensor, original_root_index_dimensions, readers);
  collector.collect();
  PatternsSet patterns = collector.access_patterns;
  AccessToPatternMap access_to_pattern_map = collector.access_to_pattern_map;

  /************* Create the cache stage *************/
  // Create the body of the cache stage
  std::string cache_name = tensor->op->name + "." + scope + suffix;
  std::string cache_tag = {};
  Map<std::string, ObjectRef> cache_attrs = {};

  Array<IterVar> cache_axis;
  {
    std::unordered_map<const VarNode*, PrimExpr> replace_map;
    for (size_t i = 0; i < original_loop_axis.size(); ++i) {
      auto lv = original_loop_axis[i];
      Var var = Var("lv" + std::to_string(i), DataType::Int(32));
      VarReplacer replacer(replace_map);
      cache_axis.push_back(IterVarNode::make(
          Range::make_by_min_extent(replacer(lv->dom->min), replacer(lv->dom->extent)), var,
          lv->iter_type, lv->thread_tag));
      replace_map[lv->var.get()] = var;
    }
    cache_axis.push_back(IterVarNode::make(Range(0, static_cast<int>(patterns.size())),
                                           Var("var", DataType::Int(32)), IterVarType::kDataPar,
                                           ""));
  }

  Array<PrimExpr> cache_shape;
  {
    std::unordered_map<const VarNode*, IntSet> dom_map;
    for (size_t i = 0; i < original_loop_axis.size(); ++i) {
      PrimExpr dim_shape = EvalSet(original_loop_axis[i]->dom->extent, dom_map).max();
      cache_shape.push_back(dim_shape);
      dom_map[original_loop_axis[i]->var.get()] = IntSet::range(original_loop_axis[i]->dom);
    }
    // Pattern dimension
    cache_shape.push_back(static_cast<int>(patterns.size()));
  }

  Array<IterVar> cache_index_variables;
  Array<UninterpFun> cache_index_expressions;
  Array<Dimension> cache_index_dimensions;
  {
    for (size_t i = 0; i < original_index_expressions.size(); ++i) {
      auto uif = original_index_expressions[i];
      cache_index_variables.push_back(IterVarNode::make(
          uif->range, Var("iv" + std::to_string(i), DataType::Int(32)), IterVarType::kDataPar, ""));
    }

    cache_index_expressions = Array<UninterpFun>(original_index_expressions);
    cache_index_dimensions = Array<Dimension>(original_index_dimensions);
  }

  Array<Dimension> cache_loop_dimensions;
  Array<Dimension> cache_root_index_dimensions;
  {
    cache_loop_dimensions = Array<Dimension>(original_loop_dimensions);
    cache_root_index_dimensions = Array<Dimension>(original_loop_dimensions);
    auto variant_dim = DimensionNode::make("variants", DimensionNode::kRangeDim);
    cache_loop_dimensions.push_back(variant_dim);
    cache_root_index_dimensions.push_back(variant_dim);
  }

  PatternsVec patterns_vec;
  for (auto pattern : patterns) {
    pattern->idx = patterns_vec.size();
    patterns_vec.push_back(pattern);
  }

  // for (auto dim : cache_index_dimensions) {
  //   std::cout << "[CROA] cche indx dim " << dim << std::endl;
  // }

  Array<PrimExpr> cache_body = {CacheBodyBuilder(tensor, original_root_index_dimensions,
                                                 patterns_vec, cache_index_variables, cache_axis,
                                                 cache_index_dimensions, cache_loop_dimensions)};

  for (auto dim : cache_index_dimensions) {
    std::cout << "[CROB] cche indx dim " << dim << std::endl;
  }

  Tensor cache =
      ComputeOpNode::make(cache_name, cache_tag, cache_attrs, cache_axis, cache_shape,
                          cache_index_variables, cache_index_expressions, cache_loop_dimensions,
                          cache_index_dimensions, cache_root_index_dimensions, cache_body)
          .output(0);

  /************* Replace reader inputs *************/
  CheckSchedule(*this, "cache_read_opaque.cc:184_start_" + tensor->op->name);
  std::unordered_map<Tensor, Tensor> vmap;
  std::unordered_map<Tensor, Tensor> rvmap;
  // std::cout << "[CRO] For " << tensor << std::endl;
  for (Operation op : readers) {
    Stage s = operator[](op);

    // N.B.: This was as below before, where original_loop_dimensions
    // is passed to ReplaceInputs for orig_idx_dims, which does not
    // make sense.
    // Operation repl_op = ReplaceInputs(s->op, &access_to_pattern_map, cache,
    // cache_root_index_dimensions, original_loop_dimensions, true);

    Operation repl_op =
        ReplaceInputs(s->op, &access_to_pattern_map, cache, cache_root_index_dimensions,
                      original_root_index_dimensions, true);
    CHECK(!repl_op.same_as(s->op))
        << "Cannot find tensor " << tensor << " in the inputs to " << repl_op;
    vmap[s->op.output(0)] = repl_op.output(0);
    rvmap[repl_op.output(0)] = s->op.output(0);
    // std::cout << "[CRO]   Replacing " << s->op << " with " << repl_op << std::endl;
    s->op = repl_op;
  }
  ReplaceDataFlow((*this)->stages, &vmap, &rvmap);
  ArrayNode* stages = (*this)->stages.CopyOnWrite();
  Stage op_stage = operator[](tensor->op);
  size_t pos = FindNodeRef(stages, op_stage);
  Stage cache_stage = Stage(cache->op);
  cache_stage.set_scope(scope);
  CHECK_LT(pos, stages->data.size());
  stages->data.insert(stages->data.begin() + pos + 1, cache_stage);
  (*this)->stage_map.Set(cache->op, cache_stage);
  // Update group
  cache_stage->group = op_stage->group;
  if (cache_stage->group.defined()) {
    ++cache_stage->group->num_child_stages;
  }

  CheckSchedule(*this, "cache_read_opaque.cc:184_end_" + tensor->op->name);
  return cache;
}
}  // namespace te
}  // namespace tvm
#undef COUT
