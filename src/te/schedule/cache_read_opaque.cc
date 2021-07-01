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
std::pair<PrimExpr, PrimExpr> CacheBodyBuilder(Tensor tensor, const Array<Operation>& readers,
                                               Array<Dimension>& original_index_dimensions,
                                               const PatternsVec& patterns_vec,
                                               Array<DimInfo>& cache_dim_infos,
                                               const Var& variant_var,
                                               const Dimension& variant_dim) {
  bool print = false;  //(tensor->op->name == "child_data");
  PrimExpr body = PrimExpr(0);
  for (size_t i = 0; i < patterns_vec.size(); ++i) {
    AccessPattern* pattern = patterns_vec[i];
    PrimExpr expr;
    Array<PrimExpr> args;

    if (print) {
      std::cout << "PATTERN " << pattern->reader_op->name << " " << std::endl;
      for (auto it : pattern->idx_dim_args) {
        std::cout << " IDX " << it.first->name << " " << it.second << " "
                  << GetRef<PrimExpr>(pattern->original_access) << " " << pattern->reader_op->name
                  << std::endl;
      }
    }

    for (const auto& orig_dim : original_index_dimensions) {
      CHECK(!orig_dim->isFunDim());
      IterVar iv = GetIterVarFromDim(orig_dim, cache_dim_infos);
      if (print) std::cout << "Arg2  " << iv << " " << orig_dim << std::endl;
      args.push_back(iv->var);
    }

    expr = CallNode::make(tensor->op->output_dtype(tensor->value_index), tensor->op->name, args,
                          CallNode::Halide, tensor->op, 0);
    body = if_then_else(variant_var == static_cast<int>(i), expr, body);
    if (print) std::cout << "BODY " << body << " " << std::endl;
  }

  PrimExpr cachePred = IntImm(DataType::Bool(), 0);
  arith::Analyzer ana;
  for (auto reader : readers) {
    if (auto compute_op = reader.as<ComputeOpNode>()) {
      std::unordered_map<const VarNode*, PrimExpr> rmap;

      for (auto di : cache_dim_infos) {
        if (di->dim == variant_dim) continue;
        if (compute_op->dim2var_maps[0].find(di->dim.as<DimensionNode>()) !=
            compute_op->dim2var_maps[0].end()) {
          rmap[compute_op->GetIterVarFromDim(0, di->dim)->var.as<VarNode>()] = di->iv->var;
        }
      }

      PrimExpr total_pred = IntImm(DataType::Bool(), 1);
      if (compute_op->pred.defined()) {
        for (const auto& p : compute_op->pred) {
          total_pred = AndNode::make(total_pred, p);
          // std::cout << "  TOTALPRED " << compute_op->name << " " << ana.Simplify(total_pred)
          // << std::endl;
        }
      }
      cachePred = OrNode::make(cachePred, VarReplacer(rmap)(total_pred));
    }
  }
  // std::cout << "CACHEPRED " << tensor->op << " " << ana.Simplify(cachePred) << std::endl;
  return std::make_pair(body, ana.Simplify(cachePred));
}

Tensor CacheReadOpaqueInternal(Schedule& sch, const Tensor& tensor, const std::string& scope,
                               const Array<Operation>& readers, const std::string& suffix) {
  CheckSchedule(sch, "cache_read_opaque.cc:184_start_" + tensor->op->name);
  bool print = false;  //(tensor->op->name == "child_data" && suffix == "");
  if (print) std::cout << "[CRO] For " << tensor << " " << tensor->op << std::endl;
  /************* Collect patterns *************/
  const ComputeOpNode* compute_op = tensor->op.as<ComputeOpNode>();
  const PlaceholderOpNode* placeholder_op = tensor->op.as<PlaceholderOpNode>();
  const SingleKernelEnvelopeOpNode* sk_op = tensor->op.as<SingleKernelEnvelopeOpNode>();
  const ScanOpNode* scan_op = tensor->op.as<ScanOpNode>();
  Array<Dimension> original_root_index_dimensions;
  Array<DimInfo> original_all_dimensions;
  if (compute_op) {
    original_root_index_dimensions = compute_op->root_index_dimensions;
    original_all_dimensions = compute_op->all_dimensions;
  } else if (placeholder_op) {
    original_root_index_dimensions = placeholder_op->self_index_dimensions;
    original_all_dimensions = placeholder_op->all_dimensions;
  } else if (sk_op) {
    CHECK_EQ(sk_op->num_outputs(), 1);
    original_root_index_dimensions = sk_op->spatial_dimensions_;
    original_all_dimensions = sk_op->GetAllDimensions();
  } else if (scan_op) {
    CHECK_EQ(scan_op->num_outputs(), 1);
    original_root_index_dimensions = scan_op->spatial_dimensions_;
    original_all_dimensions = scan_op->GetAllDimensions();
  } else {
    CHECK(false);
  }

  AccessPatternCollector collector(tensor, original_root_index_dimensions, readers);
  collector.collect();
  PatternsSet patterns = collector.access_patterns;
  AccessToPatternMap access_to_pattern_map = collector.access_to_pattern_map;

  // if (print) {
  //   std::cout << "[CRO]   Patterns " << patterns.size() << std::endl;
  //   for (auto it : access_to_pattern_map) {
  //     std::cout << "[PATTERN]    " << it.first << " " << it.second << " "
  //               << it.second->original_access << std::endl;
  //   }
  // }

  /************* Create the cache stage *************/
  // Create the body of the cache stage
  std::string cache_name = tensor->op->name + "." + scope + suffix;
  std::string cache_tag = {};
  Map<std::string, ObjectRef> cache_attrs = {};

  Array<IterVar> cache_axis;
  Array<Dimension> cache_root_index_dimensions;
  Array<DimInfo> cache_all_dimensions;
  Var variant_var;
  Dimension variant_dim;
  {
    std::unordered_map<const VarNode*, PrimExpr> replace_map;
    int i = 0;
    for (const auto& di : original_all_dimensions) {
      // if (print) std::cout << "[CRO]   OrigAllDim " << di->dim << std::endl;
      CHECK(!di->dim->isFunDim());
      auto lv = di->iv;
      Var var = Var("lv" + std::to_string(i++), DataType::Int(32));
      VarReplacer replacer(replace_map);
      IterVar cache_iv = IterVarNode::make(
          Range::make_by_min_extent(replacer(lv->dom->min), replacer(lv->dom->extent)), var,
          // lv->iter_type, lv->thread_tag);
          lv->iter_type == kLoopNestOpaque ? kDataPar : lv->iter_type, lv->thread_tag);
      cache_axis.push_back(cache_iv);
      cache_all_dimensions.push_back(
          DimInfoNode::make(di->dim, cache_iv, NullValue<UninterpFun>()));
      // if (print) std::cout << "[CRO]     Pushing" << std::endl;
      cache_root_index_dimensions.push_back(di->dim);
      replace_map[lv->var.get()] = var;
    }

    IterVar cache_variant_iv =
        IterVarNode::make(Range(0, static_cast<int>(patterns.size())),
                          Var("var", DataType::Int(32)), IterVarType::kDataPar, "");
    Dimension cache_variant_dim = DimensionNode::make("variants", DimensionNode::kRangeDim);
    cache_all_dimensions.push_back(
        DimInfoNode::make(cache_variant_dim, cache_variant_iv, NullValue<UninterpFun>()));
    cache_root_index_dimensions.push_back(cache_variant_dim);
    cache_axis.push_back(cache_variant_iv);
    variant_var = cache_variant_iv->var;
    variant_dim = cache_variant_dim;
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
  if (print) std::cout << "[CRO]   Patterns " << patterns.size() << std::endl;

  auto body_and_pred =
      CacheBodyBuilder(tensor, readers, original_root_index_dimensions, patterns_vec,
                       cache_all_dimensions, variant_var, variant_dim);

  Array<PrimExpr> cache_body = {body_and_pred.first};
  Array<PrimExpr> cache_pred = {body_and_pred.second};

  // if (print) {
  // for (auto di : cache_all_dimensions) {
  // std::cout << "[CRO] ALLDIM " << tensor << ": " << di->dim << " " << di->iv << std::endl;
  // }
  // }

  Tensor cache = ComputeOpNode::make(cache_name, cache_tag, cache_attrs, cache_axis,
                                     cache_root_index_dimensions, cache_shape, {}, {},
                                     cache_all_dimensions, cache_body, cache_pred)
                     .output(0);

  AccessPattern::Equality equals;
  for (auto repr_pattern : patterns_vec) {
    for (auto it : access_to_pattern_map) {
      AccessPattern* pattern = it.second;
      if (equals(repr_pattern, pattern)) pattern->idx = repr_pattern->idx;
    }
  }

  /************* Replace reader inputs *************/
  CheckSchedule(sch, "cache_read_opaque.cc:184_mid_" + tensor->op->name);
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
    if (print) std::cout << "[CRO]   Replacing " << s->op << " with " << repl_op << std::endl;
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

  for (auto t : cache->op->InputTensors()) {
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
  //     std::cout << "[Input] " << t->op << " " << FindNodeRef(stages, sch.operator[](t->op)) <<
  //     std::endl;
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
  auto fg = GetFeedGraph(*this, true);
  CHECK(fg.count(tensor)) << " " << tensor->op;
  Array<Operation> all_readers = fg.at(tensor);
  for (auto op : readers) {
    if (precise_readers.Contains(op)) continue;
    if (all_readers.Contains(op))
      precise_readers.push_back(op);
    else if (self->stage_map.count(op) && all_readers.Contains(self->stage_map.at(op)->op)) {
      precise_readers.push_back(op);
    } else {
      // std::cout << "[CRO] Not a reader " << op << std::endl;
    }
  }
  return CacheReadOpaqueInternal(*this, tensor, scope, precise_readers, suffix);
}

Tensor Schedule::cache_read_opaque_all_readers(const Tensor& tensor, const std::string& scope,
                                               const std::string& suffix) {
  auto fg = GetFeedGraph(*this, true);
  // std::cout << " " << tensor << std::endl;
  return CacheReadOpaqueInternal(*this, tensor, scope, fg.at(tensor), suffix);
}
}  // namespace te
}  // namespace tvm
#undef COUT
