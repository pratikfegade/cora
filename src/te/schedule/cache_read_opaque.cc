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
                          const PatternsVec& patterns_vec, Array<DimInfo>& cache_dim_infos,
                          const Var variant_loop_var) {
  PrimExpr body = PrimExpr(0);
  for (size_t i = 0; i < patterns_vec.size(); ++i) {
    AccessPattern* pattern = patterns_vec[i];
    PrimExpr expr;
    Array<PrimExpr> args;

    // for (auto it : pattern->idx_dim_args) {
    //   std::cout << "PATTERN IDX " << it.first << " " << it.second << " "
    //             << GetRef<PrimExpr>(pattern->original_access) << std::endl;
    // }

    for (const auto& orig_dim : original_index_dimensions) {
      if (orig_dim->isFunDim()) {
        // std::cout << "Looking for in pattern " << orig_dim << std::endl;
        Dimension arg_dim = pattern->idx_dim_args.at(orig_dim);
        IterVar iv = GetIterVarFromDim(arg_dim, cache_dim_infos);

        if (iv.defined()) {
        } else {
          auto entry = pattern->reader_op->GetDimVarEntry(pattern->reader_val_idx, arg_dim);
          auto reader_iv = entry.iv;
          iv = IterVarNode::make(reader_iv->dom, reader_iv->var.copy_with_suffix(""),
                                 reader_iv->iter_type, reader_iv->thread_tag);
          cache_dim_infos.push_back(DimInfoNode::make(arg_dim, iv, entry.value_expr));
        }
        args.push_back(iv->var);
        // std::cout << "Arg  " << iv << std::endl;
      } else {
        IterVar iv = GetIterVarFromDim(orig_dim, cache_dim_infos);
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
  // std::cout << "[CRO] For " << tensor << " " << tensor->op << std::endl;
  /************* Collect patterns *************/
  const ComputeOpNode* compute_op = tensor->op.as<ComputeOpNode>();
  const PlaceholderOpNode* placeholder_op = tensor->op.as<PlaceholderOpNode>();
  Array<Dimension> original_root_index_dimensions;
  Array<DimInfo> original_all_dimensions;
  if (compute_op) {
    original_root_index_dimensions = compute_op->root_index_dimensions;
    original_all_dimensions = compute_op->all_dimensions;
  } else {
    original_root_index_dimensions = placeholder_op->self_index_dimensions;
    original_all_dimensions = placeholder_op->all_dimensions;
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
  Array<Dimension> cache_root_index_dimensions;
  Array<DimInfo> cache_all_dimensions;
  Var variant_loop_var;
  {
    std::unordered_map<const VarNode*, PrimExpr> replace_map;
    int i = 0;
    for (const auto& di : original_all_dimensions) {
      // std::cout << "[CRO]   OrigAllDim " << di->dim << std::endl;
      if (di->dim->isFunDim()) {
        IterVar cache_iv =
            IterVarNode::make(di->ufun->range, Var("iv" + std::to_string(i++), DataType::Int(32)),
                              IterVarType::kDataPar, "");
        cache_all_dimensions.push_back(DimInfoNode::make(di->dim, cache_iv, di->ufun));
      } else {
        auto lv = di->iv;
        Var var = Var("lv" + std::to_string(i++), DataType::Int(32));
        VarReplacer replacer(replace_map);
        IterVar cache_iv = IterVarNode::make(
            Range::make_by_min_extent(replacer(lv->dom->min), replacer(lv->dom->extent)), var,
            lv->iter_type, lv->thread_tag);
        cache_axis.push_back(cache_iv);
        cache_all_dimensions.push_back(
            DimInfoNode::make(di->dim, cache_iv, NullValue<UninterpFun>()));
        // std::cout << "[CRO]     Pushing" << std::endl;
        cache_root_index_dimensions.push_back(di->dim);
        replace_map[lv->var.get()] = var;
      }
    }

    IterVar cache_variant_iv =
        IterVarNode::make(Range(0, static_cast<int>(patterns.size())),
                          Var("var", DataType::Int(32)), IterVarType::kDataPar, "");
    Dimension cache_variant_dim = DimensionNode::make("variants", DimensionNode::kRangeDim);
    cache_all_dimensions.push_back(
        DimInfoNode::make(cache_variant_dim, cache_variant_iv, NullValue<UninterpFun>()));
    cache_root_index_dimensions.push_back(cache_variant_dim);
    cache_axis.push_back(cache_variant_iv);
    variant_loop_var = cache_variant_iv->var;
  }

  Array<PrimExpr> cache_shape;
  {
    std::unordered_map<const VarNode*, IntSet> dom_map;
    for (const auto& di : original_all_dimensions) {
      if (di->dim->isRangeDim() || di->dim->isScanDim()) {
        PrimExpr dim_shape = EvalSet(di->iv->dom->extent, dom_map).max();
        cache_shape.push_back(dim_shape);
        dom_map[di->iv->var.get()] = IntSet::range(di->iv->dom);
      }
    }
    // Pattern dimension
    cache_shape.push_back(static_cast<int>(patterns.size()));
  }

  PatternsVec patterns_vec;
  for (auto pattern : patterns) {
    pattern->idx = patterns_vec.size();
    patterns_vec.push_back(pattern);
  }

  Array<PrimExpr> cache_body = {CacheBodyBuilder(tensor, original_root_index_dimensions,
                                                 patterns_vec, cache_all_dimensions,
                                                 variant_loop_var)};

  Tensor cache = ComputeOpNode::make(cache_name, cache_tag, cache_attrs, cache_axis,
                                     cache_root_index_dimensions, cache_shape, cache_all_dimensions,
                                     cache_body)
                     .output(0);

  /************* Replace reader inputs *************/
  CheckSchedule(*this, "cache_read_opaque.cc:184_start_" + tensor->op->name);
  std::unordered_map<Tensor, Tensor> vmap;
  std::unordered_map<Tensor, Tensor> rvmap;
  for (Operation op : readers) {
    Stage s = operator[](op);

    // N.B.: This was as below before, where original_loop_dimensions
    // is passed to ReplaceInputs for orig_idx_dims, which does not
    // make sense.
    // Operation repl_op = ReplaceInputs(s->op, &access_to_pattern_map, cache,
    // cache_root_index_dimensions, original_loop_dimensions, true);

    // std::cout << "[CRO]   Replacing " << cache_root_index_dimensions.size() << std::endl;
    Operation repl_op =
        ReplaceInputs(s->op, &access_to_pattern_map, cache, cache_root_index_dimensions,
                      original_root_index_dimensions, true);
    // std::cout << "[CRO]   Replacing " << s->op << " with " << repl_op << std::endl;
    CHECK(!repl_op.same_as(s->op))
        << "Cannot find tensor " << tensor << " in the inputs to " << repl_op;
    CHECK(!repl_op->InputTensors().Contains(tensor))
        << " Not fully replaced in " << s->op << " " << tensor;
    vmap[s->op.output(0)] = repl_op.output(0);
    rvmap[repl_op.output(0)] = s->op.output(0);
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

  // std::cout << "[CRO] Done caching " << tensor << std::endl;
  CheckSchedule(*this, "cache_read_opaque.cc:184_end_" + tensor->op->name);
  return cache;
}
}  // namespace te
}  // namespace tvm
#undef COUT
