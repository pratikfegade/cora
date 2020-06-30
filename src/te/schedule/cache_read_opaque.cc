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
  bool print = false;//(tensor->op->name == "prev_c_sum");
  PrimExpr body = PrimExpr(0);
  for (size_t i = 0; i < patterns_vec.size(); ++i) {
    AccessPattern* pattern = patterns_vec[i];
    PrimExpr expr;
    Array<PrimExpr> args;

    if (print) {
      std::cout << "PATTERN " << pattern->reader_op->name << " " << std::endl;
      for (auto it : pattern->idx_dim_args) {
	std::cout << " IDX " << it.first << " " << it.second << " "
		  << GetRef<PrimExpr>(pattern->original_access) << " " << pattern->reader_op->name
		  << std::endl;
      }
    }

    for (const auto& orig_dim : original_index_dimensions) {
      if (orig_dim->isFunDim()) {
        if (print) std::cout << " Looking for dim in pattern " << orig_dim << std::endl;
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
        if (print) std::cout << "Arg  " << iv << std::endl;
      } else {
        IterVar iv = GetIterVarFromDim(orig_dim, cache_dim_infos);
        if (print) std::cout << "Arg2  " << iv << std::endl;
        args.push_back(iv);
      }
    }

    expr = CallNode::make(tensor->op->output_dtype(tensor->value_index), tensor->op->name, args,
                          CallNode::Halide, tensor->op, 0);
    body = if_then_else(variant_loop_var == static_cast<int>(i), expr, body);
  }
  return body;
}

Tensor CacheReadOpaqueInternal(Schedule& sch, const Tensor& tensor, const std::string& scope,
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
  CheckSchedule(sch, "cache_read_opaque.cc:184_start_" + tensor->op->name);
  std::unordered_map<Tensor, Tensor> vmap;
  std::unordered_map<Tensor, Tensor> rvmap;
  sch->InvalidateCache();
  sch->InitCache();
  for (Operation op : readers) {
    Stage s;
    if (sch->Contain(op)) {
      s = sch.operator[](op);
    } else {
      s = sch->op2stage_cache_.at(op.get());
    }

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
    for (int i = 0; i < s->op->num_outputs(); ++i) {
      vmap[s->op.output(i)] = repl_op.output(i);
      rvmap[repl_op.output(i)] = s->op.output(i);
    }

    Map<FunctionRef, CacheInfo>& cacheMappings = sch->cacheTensorInfos;
    if (cacheMappings.count(s->op)) {
      cacheMappings.Set(repl_op, cacheMappings.at(s->op));
    }
    s->op = repl_op;
  }
  ReplaceDataFlow(sch->stages, sch->cacheTensorInfos, &vmap, &rvmap);
  ArrayNode* stages = sch->stages.CopyOnWrite();
  Stage op_stage = sch.operator[](tensor->op);
  size_t pos = FindNodeRef(stages, op_stage);

  for (auto t: cache->op->InputTensors()) {
    pos = std::max(pos, FindNodeRef(stages, sch.operator[](t->op)));
  }


  Stage cache_stage = Stage(cache->op);
  cache_stage.set_scope(scope);
  CHECK_LT(pos, stages->data.size());
  stages->data.insert(stages->data.begin() + pos + 1, cache_stage);
  sch->stage_map.Set(cache->op, cache_stage);
  // Update group
  cache_stage->group = op_stage->group;
  if (cache_stage->group.defined()) {
    ++cache_stage->group->num_child_stages;
  }

  // if (tensor->op->name == "prev_c_sum") {
  //   for (auto s: sch->stages) {
  //     std::cout << "[ST_ORDER] " << s->op << std::endl;
  //   }

  //   for (auto t: cache->op->InputTensors()) {
  //     std::cout << "[Input] " << t->op << " " << FindNodeRef(stages, sch.operator[](t->op)) << std::endl;
  //   }
  // }

  // Update cacheInfos
  Array<Map<Dimension, Dimension>> variantMappings;
  for (const auto& p : patterns_vec) {
    variantMappings.push_back(p->idx_dim_args);
  }
  CacheInfo info = CacheInfoNode::make(tensor->op, cache->op, variantMappings);
  sch->cacheTensorInfos.Set(cache->op, info);
  // std::cout << "[CRO] Adding to map " << cache->op << " " << info->orig << std::endl;

  // std::cout << "[CRO] Done caching " << tensor << std::endl;
  CheckSchedule(sch, "cache_read_opaque.cc:184_end_" + tensor->op->name, false);
  return cache;
}

Tensor Schedule::cache_read_opaque(const Tensor& tensor, const std::string& scope,
                                   const Array<Operation>& readers, const std::string& suffix) {
  Schedule& self = *this;
  // std::cout << "[CRO] Caching " << tensor->op << std::endl;
  Array<Operation> precise_readers;
  Array<Operation> all_readers = GetFeedGraph(*this, true).at(tensor);
  for (auto op : readers) {
    if (all_readers.Contains(op))
      precise_readers.push_back(op);
    else if (self->stage_map.count(op) && all_readers.Contains(self->stage_map.at(op)->op)) {
      precise_readers.push_back(op);
    } else {
      std::cout << "[CRO] Not a reader " << op << std::endl;
    }
  }
  return CacheReadOpaqueInternal(*this, tensor, scope, precise_readers, suffix);
}

Tensor Schedule::cache_read_opaque_all_readers(const Tensor& tensor, const std::string& scope,
                                               const std::string& suffix) {
  auto fg = GetFeedGraph(*this, true);
  return CacheReadOpaqueInternal(*this, tensor, scope, fg.at(tensor), suffix);
}
}  // namespace te
}  // namespace tvm
#undef COUT
