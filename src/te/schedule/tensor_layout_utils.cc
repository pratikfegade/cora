#include "tensor_layout_utils.h"

#include <tvm/ir/attrs.h>

namespace tvm {
namespace te {
IterVar GetIterVarFromDim(Dimension dim, Array<IterVar>& index_variables,
                          Array<IterVar>& loop_variables, Array<Dimension>& index_dimensions,
                          Array<Dimension>& loop_dimensions) {
  for (size_t i = 0; i < loop_dimensions.size(); ++i) {
    if (dim == loop_dimensions[i]) return loop_variables[i];
  }

  for (size_t i = 0; i < index_dimensions.size(); ++i) {
    if (dim == index_dimensions[i]) return index_variables[i];
  }

  return {};
}

IterVar GetIterVarFromDim(Dimension dim, Array<DimInfo>& dim_infos) {
  for (const auto& di : dim_infos) {
    if (dim == di->dim) return di->iv;
  }

  return NullValue<IterVar>();
}

size_t AccessPattern::Hasher::operator()(const AccessPattern* pattern) const {
  AttrsHash hasher;
  using std::hash;
  size_t h = 0;
  for (auto it : pattern->idx_dim_args) {
    Dimension d = it.first;
    Dimension idx = it.second;
    h += hasher(d) + hasher(idx);
  }
  return h;
}

bool AccessPattern::Equality::operator()(const AccessPattern* p1, const AccessPattern* p2) const {
  AttrsEqual equals;
  for (auto it1 : p1->idx_dim_args) {
    Dimension d1 = it1.first;
    Dimension idx1 = it1.second;
    if (p2->idx_dim_args.count(d1)) {
      Dimension idx2 = p2->idx_dim_args.at(d1);
      if (idx1 != idx2) {
        return false;
      }
    } else {
      return false;
    }
  }
  return true;
}

Dimension AccessPatternCollector::ExprAccessPatternCollector::GetDimForVar(Var var) {
  for (size_t i = var2dim_maps.size() - 1; i >= 0; --i) {
    if (var2dim_maps[i].count(var)) return var2dim_maps[i].at(var);
  }
  return NullValue<Dimension>();
}

void AccessPatternCollector::ExprAccessPatternCollector::VisitExpr_(const CallNode* op) {
  bool print = false;  //(this->tensor->op->name == "iprev_m.ila.shared.l");
  if (!op->func.defined()) ExprVisitor::VisitExpr_(op);
  if (op->func.as<OperationNode>()) {
    Tensor t = Downcast<Operation>(op->func).output(op->value_index);
    if (t->op.defined() && print)
      std::cout << "[AP] Same name access found " << GetRef<PrimExpr>(op) << " "
                << original_index_dimensions.size() << std::endl;
    if (t->op.defined() && t == this->tensor) {
      if (print)
        std::cout << "[AP] Access found " << GetRef<PrimExpr>(op) << " "
                  << original_index_dimensions.size() << std::endl;
      AccessPattern* ap = new AccessPattern();
      for (size_t i = 0; i < original_index_dimensions.size(); ++i) {
        if (print) std::cout << "[AP]   Dim " << original_index_dimensions[i] << std::endl;
        CHECK(!original_index_dimensions[i]->isFunDim());
      }

      ap->original_access = op;
      ap->reader_op = reader_op;
      ap->ufun = this->ufun;
      ap->reader_val_idx = this->reader_val_idx;
      this->access_patterns->insert(ap);
      (*this->access_to_pattern_map)[op] = ap;
    }
  } else if (auto curr_ufun = op->func.as<UninterpFunNode>()) {
    // std::cout << "[AP]   UF " << GetRef<PrimExpr>(op) << std::endl;
    const UninterpFunNode* old_ufun;
    std::swap(this->ufun, old_ufun);
    this->ufun = curr_ufun;

    Map<Var, Dimension> ufun_var2dim_map;
    for (size_t i = 0; i < ufun->dimensions.size(); ++i) {
      ufun_var2dim_map.Set(ufun->parameters[i], ufun->dimensions[i]);
    }
    var2dim_maps.push_back(ufun_var2dim_map);
    this->operator()(ufun->body);
    var2dim_maps.resize(var2dim_maps.size() - 1);
    std::swap(this->ufun, old_ufun);
  }
  ExprVisitor::VisitExpr_(op);
}

void AccessPatternCollector::ExprAccessPatternCollector::collect(const UninterpFunNode* ufun,
                                                                 Map<Var, Dimension> var2dim_map_,
                                                                 int reader_val_idx_) {
  this->var2dim_maps.push_back(var2dim_map_);
  this->ufun = ufun;
  this->reader_val_idx = reader_val_idx_;
  this->operator()(ufun->body);
}

void AccessPatternCollector::ExprAccessPatternCollector::collect(PrimExpr expr,
                                                                 Map<Var, Dimension> var2dim_map_,
                                                                 int reader_val_idx_) {
  this->var2dim_maps.push_back(var2dim_map_);
  this->ufun = nullptr;
  this->reader_val_idx = reader_val_idx_;
  this->operator()(expr);
}

void AccessPatternCollector::collect() {
  for (auto reader : this->readers) {
    // std::cout << "[AP] Collecting patterns in " << reader << " " << this->tensor << std::endl;
    if (auto reader_op = reader.as<ComputeOpNode>()) {
      ExprAccessPatternCollector exprCollector(this->tensor, original_index_dimensions,
                                               &(this->access_patterns),
                                               &(this->access_to_pattern_map), reader_op);
      Map<Var, Dimension> op_var2dim_map;
      for (const auto& it : reader_op->var2dim_map) {
        op_var2dim_map.Set(GetRef<Var>(it.first), GetRef<Dimension>(it.second));
      }
      for (const auto& body_expr : reader_op->body) {
        exprCollector.collect(body_expr, op_var2dim_map, 0);
      }
      for (const auto& di : reader_op->all_dimensions) {
        CHECK(!di->dim->isFunDim());
        // std::cout << "[AP]   Extent " << di->iv->dom->extent << std::endl;
        exprCollector.collect(di->iv->dom->extent, op_var2dim_map, 0);
      }
    } else if (auto reader_op = reader.as<ScanOpNode>()) {
      ExprAccessPatternCollector exprCollector(this->tensor, original_index_dimensions,
                                               &(this->access_patterns),
                                               &(this->access_to_pattern_map), reader_op);

      Map<Var, Dimension> var2dim_map;
      for (const auto& dim2var_map : reader_op->dim2var_maps) {
        for (const auto& it : dim2var_map) {
          var2dim_map.Set(it.second.iv->var, GetRef<Dimension>(it.first));
        }
      }
      for (int i = 0; i < reader_op->num_outputs(); ++i) {
        for (const auto& it : reader_op->dim2var_maps[i]) {
          CHECK(!it.first->isFunDim());
          exprCollector.collect(it.second.iv->dom->extent, var2dim_map, i);
        }
      }
    } else if (auto reader_op = reader.as<SingleKernelEnvelopeOpNode>()) {
      ExprAccessPatternCollector exprCollector(this->tensor, original_index_dimensions,
                                               &(this->access_patterns),
                                               &(this->access_to_pattern_map), reader_op);

      Map<Var, Dimension> var2dim_map;
      for (const auto& dim2var_map : reader_op->dim2var_maps) {
        for (const auto& it : dim2var_map) {
          var2dim_map.Set(it.second.iv->var, GetRef<Dimension>(it.first));
        }
      }
      for (int i = 0; i < reader_op->num_outputs(); ++i) {
        for (const auto& it : reader_op->dim2var_maps[i]) {
          CHECK(!it.first->isFunDim());
          exprCollector.collect(it.second.iv->dom->extent, var2dim_map, i);
        }
      }
    } else if (auto reader_op = reader.as<ConditionalOpNode>()) {
      ExprAccessPatternCollector exprCollector(this->tensor, original_index_dimensions,
                                               &(this->access_patterns),
                                               &(this->access_to_pattern_map), reader_op);

      Map<Var, Dimension> var2dim_map;
      for (const auto& dim2var_map : reader_op->dim2var_maps) {
        for (const auto& it : dim2var_map) {
          var2dim_map.Set(it.second.iv->var, GetRef<Dimension>(it.first));
        }
      }
      for (int i = 0; i < reader_op->num_outputs(); ++i) {
        for (const auto& it : reader_op->dim2var_maps[i]) {
          CHECK(!it.first->isFunDim());
          exprCollector.collect(it.second.iv->dom->extent, var2dim_map, i);
        }
      }
      exprCollector.collect(reader_op->condition, var2dim_map, 0);
    } else {
      CHECK(false) << "Opaque caching is not yet implemented for reader op " << reader;
    }
  }
}

Operation ReplaceInputs(Operation reader, const AccessToPatternMap* patterns_map, Tensor cache,
                        Array<Dimension> cache_idx_dims, Array<Dimension> orig_idx_dims,
                        bool add_variant_dimension) {
  class AbstractReplacer : public ExprMutator {
    PrimExpr VisitExpr_(const CallNode* op) override {
      if (this->patterns_map->find(op) != this->patterns_map->end()) {
        auto pattern = this->patterns_map->find(op)->second;
        // std::cout << "[RI]    Found call " << GetRef<PrimExpr>(op) << " " << op << " " <<
        // pattern
        //           << " " << pattern->original_access << std::endl;
        Array<PrimExpr> args;
        // Skip the last dimension as that's the variant dimension
        // we handle after the loop
        for (size_t i = 0;
             i < (add_variant_dimension ? cache_idx_dims.size() - 1 : cache_idx_dims.size()); ++i) {
          auto dim = cache_idx_dims[i];
          PrimExpr arg;
          if (!orig_idx_dims.Contains(dim)) {
            // This is a newly added dimension, corresponding to an
            // index of the original tensor. For this, we need to
            // index by the IV corresponding to this dimension.
            arg = this->GetVarFromNewlyAddedDimension(pattern, dim);
            // std::cout << "[RI] Dim1 " << dim << " " << arg << std::endl;
          } else {
            // Here we leave the argument intact, for the case
            // where the dimension is left unmodified by the
            // transform.

            // N.B. This was "arg = op->args[i];" earlier, but was
            // changed into the following as i may not correspond to
            // the correct argument in the call as a single cache
            // dimension may expand to multiple original dimensions.
            arg = op->args[orig_idx_dims.GetIdx(dim)];

            // std::cout << "[RI] Dim2 " << dim << " " << arg << std::endl;
          }
          args.push_back(arg);
        }
        if (add_variant_dimension) {
          // std::cout << "[RI] Variant " << pattern->idx << std::endl;
          args.push_back(pattern->idx);
        }
        PrimExpr new_call =
            CallNode::make(op->dtype, this->cache->op->name, args, op->call_type, op->arg_dims,
                           this->cache->op, this->cache->value_index, op->custom_realize_bounds);
        if (args.size() == 0) {
          // std::cout << "[RI] Returning " << new_call << std::endl;
          for (auto dim : cache_idx_dims) {
            // std::cout << "[RI]     Dim " << dim << std::endl;
          }
        }
        return new_call;
      } else
        return ExprMutator::VisitExpr_(op);
    }

    virtual Var GetVarFromNewlyAddedDimension(const AccessPattern* pattern,
                                              const Dimension& dim) = 0;

   public:
    AbstractReplacer(const AccessToPatternMap* patterns_map_, Tensor cache_,
                     Array<Dimension> cache_idx_dims_, Array<Dimension> orig_idx_dims_,
                     bool add_variant_dimension_)
        : patterns_map(patterns_map_),
          cache(cache_),
          cache_idx_dims(cache_idx_dims_),
          orig_idx_dims(orig_idx_dims_),
          add_variant_dimension(add_variant_dimension_) {
      if (cache_idx_dims.size() == 0) {
        // std::cout << "[CRO]    Replacer " << cache_idx_dims.size() << std::endl;
      }
    }

    const AccessToPatternMap* patterns_map;
    Tensor cache;
    Array<Dimension> cache_idx_dims;
    Array<Dimension> orig_idx_dims;
    bool add_variant_dimension;
  };

  class ExprReplacer : public AbstractReplacer {
    Var GetVarFromNewlyAddedDimension(const AccessPattern* pattern, const Dimension& dim) override {
      return vardim_op->GetIterVarFromDim(pattern->reader_val_idx, dim)->var;
    }

   public:
    ExprReplacer(const BaseVarDimOpNode* vardim_op_, const AccessToPatternMap* patterns_map_,
                 Tensor cache_, Array<Dimension> cache_idx_dims_, Array<Dimension> orig_idx_dims_,
                 bool add_variant_dimension_)
        : AbstractReplacer(patterns_map_, cache_, cache_idx_dims_, orig_idx_dims_,
                           add_variant_dimension_),
          vardim_op(vardim_op_) {
      if (cache_idx_dims_.size() == 0) {
        // std::cout << "[CRO]    ExprReplacer " << cache_idx_dims_.size() << std::endl;
      }
    }

    const BaseVarDimOpNode* vardim_op;
  };

  class UFReplacer : public AbstractReplacer {
    Var GetVarFromNewlyAddedDimension(const AccessPattern* pattern, const Dimension& dim) override {
      if (orig->dimensions.Contains(dim)) {
        return orig->parameters[orig->dimensions.GetIdx(dim)];
      } else {
        new_param_dims.push_back(dim);
        Var new_param = Var("p" + dim->name, DataType::Int(32));
        new_params.push_back(new_param);
        return new_param;
      }
    }

   public:
    UFReplacer(const AccessToPatternMap* patterns_map_, Tensor cache_,
               Array<Dimension> cache_idx_dims_, Array<Dimension> orig_idx_dims_,
               bool add_variant_dimension_)
        : AbstractReplacer(patterns_map_, cache_, cache_idx_dims_, orig_idx_dims_,
                           add_variant_dimension_) {}

    UninterpFun replace(UninterpFun orig_) {
      this->orig = orig_;
      this->new_param_dims.resize(0);
      this->new_params.resize(0);

      PrimExpr body = this->VisitExpr(orig->body);
      if (body.same_as(orig->body)) {
        return orig;
      }
      Array<Var> parameters = Array<Var>(orig->parameters);
      Array<Dimension> dimensions = Array<Dimension>(orig->dimensions);
      for (size_t i = 0; i < new_params.size(); ++i) {
        parameters.push_back(new_params[i]);
        dimensions.push_back(new_param_dims[i]);
      }
      return UninterpFunNode::make(orig->fname + ".r", orig->range, dimensions, parameters, body,
                                   orig->type);
    }

    UninterpFun orig;
    Array<Dimension> new_param_dims;
    Array<Var> new_params;
  };

  class Replacer : public ExprMutator {
    PrimExpr VisitExpr_(const CallNode* op) override {
      bool print = false;  //(vardim_op->name == "imml.ila.rf");
      if (this->patterns_map->find(op) != this->patterns_map->end()) {
        if (print) std::cout << "[RI] Found call " << GetRef<PrimExpr>(op) << std::endl;
        auto pattern = this->patterns_map->find(op)->second;
        Array<PrimExpr> args;
        // Skip the last dimension as that's the variant dimension
        // we handle after the loop
        for (size_t i = 0;
             i < (add_variant_dimension ? cache_idx_dims.size() - 1 : cache_idx_dims.size()); ++i) {
          auto dim = cache_idx_dims[i];
          PrimExpr arg;
          if (!orig_idx_dims.Contains(dim)) {
            // This is a newly added dimension, corresponding to an
            // index of the original tensor. For this, we need to
            // index by the IV corresponding to this dimension.
            arg = this->GetVarFromNewlyAddedDimension(pattern, dim);
            // std::cout << "[RI] Dim1 " << dim << " " << arg << std::endl;
          } else {
            // Here we leave the argument intact, for the case
            // where the dimension is left unmodified by the
            // transform.

            // N.B. This was "arg = op->args[i];" earlier, but was
            // changed into the following as i may not correspond to
            // the correct argument in the call as a single cache
            // dimension may expand to multiple original dimensions.
            arg = op->args[orig_idx_dims.GetIdx(dim)];

            // std::cout << "[RI] Dim2 " << dim << " " << arg << std::endl;
          }
          args.push_back(arg);
        }
        if (add_variant_dimension) {
          args.push_back(pattern->idx);
        }
        PrimExpr new_call =
            CallNode::make(op->dtype, this->cache->op->name, args, op->call_type, op->arg_dims,
                           this->cache->op, this->cache->value_index, op->custom_realize_bounds);
        // std::cout << "[RI]   Returning " << new_call << std::endl;
        return new_call;
      } else if (op->func.as<UninterpFunNode>()) {
        // if (print) std::cout << "[REPLACING]  " << GetRef<PrimExpr>(op) << " " << op <<
        // std::endl;
        UninterpFun old_fun = Downcast<UninterpFun>(op->func);
        UninterpFun new_fun = replaceUf(old_fun);

        bool changed = !new_fun.same_as(old_fun);
        Array<PrimExpr> new_args;
        for (const auto& arg : op->args) {
          PrimExpr new_arg = this->VisitExpr(arg);
          if (!arg.same_as(new_arg)) changed = true;
          new_args.push_back(new_arg);
        }

        Array<Range> new_custom_realize_bounds;
        for (const auto& bound : op->custom_realize_bounds) {
          auto new_bound = Range::make_by_min_extent(this->VisitExpr(bound->min),
                                                     this->VisitExpr(bound->extent));
          if (!bound.same_as(new_bound)) changed = true;
          new_custom_realize_bounds.push_back(new_bound);
        }

        if (changed)
          return CallNode::make(op->dtype, op->name, new_args, op->call_type, op->arg_dims, new_fun,
                                op->value_index, new_custom_realize_bounds);
        else
          return GetRef<PrimExpr>(op);
      } else {
        return ExprMutator::VisitExpr_(op);
      }
    }

    Var GetVarFromNewlyAddedDimension(const AccessPattern* pattern, const Dimension& dim) {
      if (this->orig.defined()) {
        if (orig->dimensions.Contains(dim)) {
          return orig->parameters[orig->dimensions.GetIdx(dim)];
        } else {
          new_param_dims.push_back(dim);
          Var new_param = Var("p" + dim->name, DataType::Int(32));
          new_params.push_back(new_param);
          return new_param;
        }
      } else {
        return vardim_op->GetIterVarFromDim(pattern->reader_val_idx, dim)->var;
      }
    }

   public:
    UninterpFun replaceUf(UninterpFun orig_) {
      // bool print = (vardim_op->name == "css_update");
      UninterpFun old_orig;
      Array<Dimension> old_new_param_dims;
      Array<Var> old_new_params;

      std::swap(this->orig, old_orig);
      std::swap(this->new_param_dims, old_new_param_dims);
      std::swap(this->new_params, old_new_params);

      this->orig = orig_;
      this->new_param_dims = Array<Dimension>();
      this->new_params = Array<Var>();

      // if (print) std::cout << "[UFREPL]  " << orig->body << std::endl;
      PrimExpr body = this->VisitExpr(orig->body);
      // if (print) std::cout << "[UFREPL]  " << body << std::endl;
      UninterpFun ret = orig;
      if (!body.same_as(orig->body)) {
        Array<Var> parameters = Array<Var>(orig->parameters);
        Array<Dimension> dimensions = Array<Dimension>(orig->dimensions);
        for (size_t i = 0; i < new_params.size(); ++i) {
          parameters.push_back(new_params[i]);
          dimensions.push_back(new_param_dims[i]);
        }
        ret = UninterpFunNode::make(orig->fname + ".r", orig->range, dimensions, parameters, body,
                                    orig->type);
      }
      std::swap(this->orig, old_orig);
      std::swap(this->new_param_dims, old_new_param_dims);
      std::swap(this->new_params, old_new_params);
      // std::cout << "[UFREPLRET]  " << ret->body << std::endl;
      return ret;
    }

    Replacer(const AccessToPatternMap* patterns_map_, Tensor cache_,
             Array<Dimension> cache_idx_dims_, Array<Dimension> orig_idx_dims_,
             bool add_variant_dimension_, const BaseVarDimOpNode* vardim_op_)
        : patterns_map(patterns_map_),
          cache(cache_),
          cache_idx_dims(cache_idx_dims_),
          orig_idx_dims(orig_idx_dims_),
          add_variant_dimension(add_variant_dimension_),
          vardim_op(vardim_op_) {}

    const AccessToPatternMap* patterns_map;
    Tensor cache;
    Array<Dimension> cache_idx_dims;
    Array<Dimension> orig_idx_dims;
    bool add_variant_dimension;

    const BaseVarDimOpNode* vardim_op;

    UninterpFun orig = NullValue<UninterpFun>();
    Array<Dimension> new_param_dims;
    Array<Var> new_params;
  };

  if (auto compute_op = reader.as<ComputeOpNode>()) {
    auto new_op = make_object<ComputeOpNode>(*compute_op);
    bool print = false;  //(compute_op->name == "imml.ila.rf");
    if (print) std::cout << "[RI] Replacing in " << compute_op->name << std::endl;
    bool changed = false;
    ExprReplacer expr_replacer(compute_op, patterns_map, cache, cache_idx_dims, orig_idx_dims,
                               add_variant_dimension);

    Array<PrimExpr> arr;
    if (compute_op->body[0]->IsInstance<tir::ReduceNode>()) {
      // Specially handle reduce so the replaced op
      // still share all the components
      PrimExpr new_reduce = expr_replacer(compute_op->body[0]);
      if (print) std::cout << "[RI]  Body replaced to " << new_reduce << std::endl;
      if (!new_reduce.same_as(compute_op->body[0])) {
        const tir::ReduceNode* r = new_reduce.as<tir::ReduceNode>();
        for (size_t k = 0; k < compute_op->body.size(); ++k) {
          auto n = make_object<tir::ReduceNode>(*r);
          n->value_index = static_cast<int>(k);
          n->dtype = r->source[k].dtype();
          arr.push_back(PrimExpr(n));
        }
      } else {
        arr = compute_op->body;
      }
    } else {
      for (auto e : compute_op->body) {
        PrimExpr new_expr = expr_replacer(e);
        if (print) std::cout << "[RI]  Body replaced to " << new_expr << std::endl;
        arr.push_back(new_expr);
      }
    }

    Array<PrimExpr> pred_arr;
    for (auto e : compute_op->pred) {
      PrimExpr new_expr = expr_replacer(e);
      if (print) std::cout << "[RI]  Replaced to " << new_expr << std::endl;
      pred_arr.push_back(new_expr);
    }

    UFReplacer uf_replacer(patterns_map, cache, cache_idx_dims, orig_idx_dims,
                           add_variant_dimension);
    Replacer new_replacer(patterns_map, cache, cache_idx_dims, orig_idx_dims, add_variant_dimension,
                          compute_op);

    Array<DimInfo> new_dim_infos;
    for (const auto di : new_op->all_dimensions) {
      CHECK(!di->dim->isFunDim());
      IterVar iv = di->iv;
      PrimExpr old_extent = iv->dom->extent;
      PrimExpr new_extent = new_replacer(old_extent);
      if (!new_extent.same_as(old_extent)) {
        if (print)
          std::cout << "[REPL]  Extent " << UninterpFun::InlineUninterpFunCalls(old_extent) << " "
                    << UninterpFun::InlineUninterpFunCalls(new_extent) << std::endl;
        const_cast<RangeNode*>(iv->dom.as<RangeNode>())->extent = new_extent;
        changed = true;
      }
      // if (print) std::cout << "[REPL] " << di->iv << std::endl;
      new_dim_infos.push_back(DimInfoNode::make(di->dim, di->iv));
    }
    new_op->set_all_dimensions(new_dim_infos);

    if (!arr.same_as(compute_op->body)) {
      new_op->body = arr;
      changed = true;
    }

    if (!pred_arr.same_as(compute_op->pred)) {
      new_op->pred = pred_arr;
      changed = true;
    }

    if (changed) {
      new_op->RefreshDimVarMappings();
      new_op->set_realize_bounds(compute_op->realize_bounds, compute_op->who_set_realize_bounds);
      if (print) std::cout << "[REPL] Returning new" << std::endl;
      return Operation(new_op);
    } else
      return reader;
  } else if (auto scan_op = reader.as<ScanOpNode>()) {
    // std::cout << "[REPL] OP " << reader << std::endl;
    auto new_op = make_object<ScanOpNode>(*scan_op);
    bool changed = false;
    UFReplacer uf_replacer(patterns_map, cache, cache_idx_dims, orig_idx_dims,
                           add_variant_dimension);
    Replacer new_replacer(patterns_map, cache, cache_idx_dims, orig_idx_dims, add_variant_dimension,
                          scan_op);

    std::vector<std::unordered_map<const DimensionNode*, DimVarEntry>> new_dim2var_maps;
    for (auto& dim2var_map : new_op->dim2var_maps) {
      std::unordered_map<const DimensionNode*, DimVarEntry> new_dim2var_map;
      for (auto& it : dim2var_map) {
        IterVar iv = it.second.iv;
        PrimExpr old_extent = iv->dom->extent;
        PrimExpr new_extent = new_replacer(old_extent);
        if (!new_extent.same_as(old_extent)) {
          // std::cout << "[REPL]   Extent " << old_extent << " " << new_extent << " " << it.first
          //           << std::endl;
          const_cast<RangeNode*>(iv->dom.as<RangeNode>())->extent = new_extent;
          changed = true;
        }

        CHECK(!it.first->isFunDim());
        new_dim2var_map[it.first] = {it.second.dim, it.second.iv};
      }
      new_dim2var_maps.push_back(new_dim2var_map);
    }
    new_op->dim2var_maps = new_dim2var_maps;

    if (changed) {
      Operation op = Operation(new_op);
      for (auto t : op->InputTensors()) {
        // std::cout << "[REPL]  Input tensor " << t << std::endl;
      }
      return op;
    } else
      return reader;
  } else if (auto sk_op = reader.as<SingleKernelEnvelopeOpNode>()) {
    // std::cout << "[REPL] OP " << reader << std::endl;
    auto new_op = make_object<SingleKernelEnvelopeOpNode>(*sk_op);
    bool changed = false;
    UFReplacer uf_replacer(patterns_map, cache, cache_idx_dims, orig_idx_dims,
                           add_variant_dimension);
    Replacer new_replacer(patterns_map, cache, cache_idx_dims, orig_idx_dims, add_variant_dimension,
                          sk_op);

    std::vector<std::unordered_map<const DimensionNode*, DimVarEntry>> new_dim2var_maps;
    for (auto& dim2var_map : new_op->dim2var_maps) {
      std::unordered_map<const DimensionNode*, DimVarEntry> new_dim2var_map;
      for (auto& it : dim2var_map) {
        IterVar iv = it.second.iv;
        PrimExpr old_extent = iv->dom->extent;
        PrimExpr new_extent = new_replacer(old_extent);
        if (!new_extent.same_as(old_extent)) {
          // std::cout << "[REPL]   Extent " << iv << " "
          //           << UninterpFun::InlineUninterpFunCalls(old_extent) << " "
          //           << UninterpFun::InlineUninterpFunCalls(new_extent) << " " << it.first
          //           << std::endl;
          const_cast<RangeNode*>(iv->dom.as<RangeNode>())->extent = new_extent;
          changed = true;
        }

        CHECK(!it.first->isFunDim());
        new_dim2var_map[it.first] = {it.second.dim, it.second.iv};
      }
      new_dim2var_maps.push_back(new_dim2var_map);
    }
    new_op->dim2var_maps = new_dim2var_maps;

    if (changed) {
      Operation op = Operation(new_op);
      for (auto t : op->InputTensorsWithUnemitted()) {
        // std::cout << "[REPL]  Input tensor " << t << std::endl;
      }
      return op;
    } else
      return reader;
  } else if (auto conditional_op = reader.as<ConditionalOpNode>()) {
    // std::cout << "[REPL] OP " << reader << std::endl;
    auto new_op = make_object<ConditionalOpNode>(*conditional_op);
    bool changed = false;
    UFReplacer uf_replacer(patterns_map, cache, cache_idx_dims, orig_idx_dims,
                           add_variant_dimension);
    Replacer new_replacer(patterns_map, cache, cache_idx_dims, orig_idx_dims, add_variant_dimension,
                          conditional_op);

    std::vector<std::unordered_map<const DimensionNode*, DimVarEntry>> new_dim2var_maps;
    for (auto& dim2var_map : new_op->dim2var_maps) {
      std::unordered_map<const DimensionNode*, DimVarEntry> new_dim2var_map;
      for (auto& it : dim2var_map) {
        IterVar iv = it.second.iv;
        PrimExpr old_extent = iv->dom->extent;
        PrimExpr new_extent = new_replacer(old_extent);
        if (!new_extent.same_as(old_extent)) {
          // std::cout << "[REPL]   Extent " << old_extent << " " << new_extent << " " << it.first
          //           << std::endl;
          const_cast<RangeNode*>(iv->dom.as<RangeNode>())->extent = new_extent;
          changed = true;
        }

        CHECK(!it.first->isFunDim());
        new_dim2var_map[it.first] = {it.second.dim, it.second.iv};
      }
      new_dim2var_maps.push_back(new_dim2var_map);
    }
    new_op->dim2var_maps = new_dim2var_maps;

    ExprReplacer expr_replacer(compute_op, patterns_map, cache, cache_idx_dims, orig_idx_dims,
                               add_variant_dimension);
    PrimExpr new_condition = expr_replacer(conditional_op->condition);
    if (!new_condition.same_as(conditional_op->condition)) {
      changed = true;
      new_op->condition = new_condition;
    }

    if (changed) {
      Operation op = Operation(new_op);
      for (auto t : op->InputTensors()) {
        // std::cout << "[REPL]  Input tensor " << t << std::endl;
      }
      return op;
    } else
      return reader;
  } else {
    CHECK(false) << "Only scan and compute readers supported";
    return reader;
  }
}

PrimExpr lower_tensor_access(std::string name, Stage s, Array<PrimExpr> coords,
                             std::unordered_map<const DimensionNode*, PrimExpr> full_coords,
                             Modes root_layout, Modes leaf_layout) {
  if (!s.is_ancestor_attached_at_root()) return {};
  if (name != "B") return {};

  CHECK_EQ(coords.size(), leaf_layout->dimensions.size());

  std::cout << "[LTA] For " << s << std::endl;
  DimDepMap outer_to_inner_deps;
  DimDepMap inner_to_outer_deps;
  LeafDimensionsDependenceInformation(s, root_layout, &outer_to_inner_deps, &inner_to_outer_deps);

  {
    std::cout << "[LTA]  Leaf dim deps computed" << std::endl;
    for (auto it : outer_to_inner_deps) {
      std::cout << "[LTA]   Outer dim " << it.first->name << std::endl;
      for (auto d : it.second) {
        std::cout << "[LTA]    Inner dim " << d->name << std::endl;
      }
    }

    std::cout << "[LTA]  Root layout" << std::endl;
    for (size_t i = 0; i < root_layout->dimensions.size(); ++i) {
      std::cout << "[LTA]   Dim " << root_layout->dimensions[i] << " " << root_layout->a_funs[i]
                << std::endl;
    }

    std::cout << "[LTA]  Leaf layout" << std::endl;
    for (size_t i = 0; i < leaf_layout->dimensions.size(); ++i) {
      std::cout << "[LTA]   Dim " << leaf_layout->dimensions[i] << " " << leaf_layout->a_funs[i]
                << std::endl;
    }
  }

  PrimExpr offset = 0;

  Array<Dimension> full_coords_dims;
  Array<PrimExpr> full_coords_coords;
  for (auto it : full_coords) {
    full_coords_dims.push_back(GetRef<Dimension>(it.first));
    full_coords_coords.push_back(it.second);
  }

  std::function<void(const DimensionNode*, DimNodeSet&)> get_transitive_dependent_dims;
  get_transitive_dependent_dims = [&](const DimensionNode* dim, DimNodeSet& res) {
    for (auto dep_dim : outer_to_inner_deps[dim]) {
      res.insert(dep_dim);
      get_transitive_dependent_dims(dep_dim, res);
    }
    return;
  };

  std::map<std::set<const DimensionNode*>, UninterpFun> root_a_funs;
  for (size_t i = 0; i < root_layout->dimensions.size(); ++i) {
    if (root_layout->a_funs[i].defined()) {
      auto dep_dims = root_layout->get_transitive_dependent_dims(i);
      std::set<const DimensionNode*> key_set;
      for (auto dd : dep_dims) {
        key_set.insert(dd.operator->());
      }
      root_a_funs[key_set] = root_layout->a_funs[i];
    }
  }

  auto leaf_dimensions = leaf_layout->dimensions;
  std::cout << "[LTA]  Offset computation" << std::endl;
  DimNodeSet processed;

  for (int i = leaf_dimensions.size() - 1; i >= 0; --i) {
    Dimension outer_dim = leaf_dimensions[i];
    std::cout << "[LTA]   Outer dim " << outer_dim << std::endl;

    DimNodeSet handled_already;
    auto create_a_fun_call = [&](int dim_idx) {
      Dimension inner_dim = leaf_dimensions[dim_idx];
      std::cout << "[LTA]     AFun call" << std::endl;

      std::unordered_map<const DimensionNode*, Range> pdd_state;
      for (size_t k = 0; k < leaf_dimensions.size(); ++k) {
        auto ld = leaf_dimensions[k];
        auto ldn = ld.operator->();
        if (processed.count(ldn)) {
          pdd_state[ldn] = Range::make_by_min_max_exclusive(
              0, leaf_layout->l_funs[k].MakeCallTo(full_coords_coords, full_coords_dims));
        } else {
          pdd_state[ldn] = Range::make_by_min_extent(coords[k], 1);
        }
      }
      DimensionPassUpDomain(s, &pdd_state, false);

      std::cout << "[LTA]      All domains" << std::endl;
      for (auto rd : root_layout->dimensions) {
        std::cout << "[LTA]       " << rd->name << " " << pdd_state[rd.operator->()] << std::endl;
      }

      std::cout << "[LTA]      Transitive ragged leaf domains" << std::endl;
      DimNodeSet transitive_dependent_dims;
      get_transitive_dependent_dims(inner_dim.operator->(), transitive_dependent_dims);
      for (auto dd : transitive_dependent_dims) {
        std::cout << "[LTA]       TDD " << dd->name << std::endl;
      }

      handled_already.insert(transitive_dependent_dims.begin(), transitive_dependent_dims.end());

      UninterpFun a_fun_to_call = NullValue<UninterpFun>();

      std::unordered_set<const DimensionNode*> state;
      bool check = true;
      for (auto dim : transitive_dependent_dims) {
        state.insert(dim);
      }
      DimensionPassUpBitMaskExact(s, &state, &check);
      CHECK(check);

      std::set<const DimensionNode*> key;
      for (auto rd : root_layout->dimensions) {
        if (state.count(rd.operator->())) {
          key.insert(rd.operator->());
        }
      }

      CHECK(root_a_funs.count(key));
      std::cout << "[LTA]      AFun found " << root_a_funs[key] << std::endl;
      a_fun_to_call = root_a_funs[key];

      int num_range_dimensions;
      Array<PrimExpr> call1_args;
      Array<PrimExpr> call2_args;
      CHECK(a_fun_to_call->dimensions.defined());
      for (auto rd : a_fun_to_call->dimensions) {
        CHECK(pdd_state.count(rd.operator->()));
        Range r = pdd_state.at(rd.operator->());
        if (is_one(Simplify(r->extent))) {
          call1_args.push_back(r->min);
          call2_args.push_back(r->min);
        } else {
          call1_args.push_back(r->max_exclusive());
          call2_args.push_back(r->min);
          num_range_dimensions++;
        }
      }

      PrimExpr contribution = (a_fun_to_call.MakeCallTo(call1_args, a_fun_to_call->dimensions) -
                               a_fun_to_call.MakeCallTo(call2_args, a_fun_to_call->dimensions));
      return contribution;
    };

    PrimExpr t_expr = 0;
    if (outer_to_inner_deps.at(outer_dim.operator->()).size() == 0) {
      PrimExpr contribution =
          leaf_layout->l_funs[i].MakeCallTo(full_coords_coords, full_coords_dims);
      std::cout << "[LTA]     Simple width "
                << " " << leaf_layout->l_funs[i] << " " << contribution << std::endl;
      t_expr = coords[i];
    } else {
      std::cout << "[LTA]     AFun call" << std::endl;

      PrimExpr contribution = create_a_fun_call(i);
      std::cout << "[LTA]       Contribution " << contribution << std::endl;
      t_expr = t_expr + contribution;
    }

    for (int j = i + 1; j < leaf_dimensions.size(); ++j) {
      Dimension inner_dim = leaf_dimensions[j];
      if (handled_already.count(inner_dim.operator->())) {
        continue;
      }
      std::cout << "[LTA]    Inner dim " << inner_dim << std::endl;
      DimNodeSet inner_deps_of_inner_dim = outer_to_inner_deps.at(inner_dim.operator->());
      DimNodeSet outer_deps_of_inner_dim = inner_to_outer_deps.at(inner_dim.operator->());

      bool all_outer_deps_unrelaxed = true;
      for (auto d_node : outer_deps_of_inner_dim) {
        if (processed.count(d_node)) {
          all_outer_deps_unrelaxed = false;
          break;
        }
      }

      if (inner_deps_of_inner_dim.size() == 0 && all_outer_deps_unrelaxed) {
        PrimExpr contribution =
            leaf_layout->l_funs[j].MakeCallTo(full_coords_coords, full_coords_dims);
        std::cout << "[LTA]     Simple width "
                  << " " << leaf_layout->l_funs[j] << " " << contribution << std::endl;
        t_expr = t_expr * contribution;
      } else if (inner_deps_of_inner_dim.size() > 0) {
        std::cout << "[LTA]     AFun call" << std::endl;

        PrimExpr contribution = create_a_fun_call(j);
        std::cout << "[LTA]       Contribution " << contribution << std::endl;
        t_expr = t_expr + contribution;
      } else if (!all_outer_deps_unrelaxed) {
        CHECK(false);
      }
    }

    offset = offset + t_expr;
    processed.insert(outer_dim.operator->());
  }

  offset = Simplify(offset);
  std::cout << "[LTA]   Offset " << Simplify(UninterpFun::InlineUninterpFunCalls(offset))
            << std::endl;
  return offset;
}

Operation ReplaceInputsGeneral(Stage s, Operation old_op, Operation repl_op, Operation reader,
                               const Map<IterVar, Range>& dom_map, Array<Modes> root_layouts) {
  class Replacer : public ExprMutator {
    PrimExpr VisitExpr_(const CallNode* op) override {
      bool print = false;  //(op->name == "A.shared.r");
      if (op->call_type == CallNode::Halide && op->func == old_op) {
        if (print) {
          std::cout << "[RIG]  Replacing access " << GetRef<PrimExpr>(op) << std::endl;
        }
        std::unordered_map<const DimensionNode*, PrimExpr> state;
        CHECK_EQ(s->dim_relation_graph->root_dimensions.size(), op->args.size());
        for (size_t i = 0; i < s->dim_relation_graph->root_dimensions.size(); ++i) {
          state[s->dim_relation_graph->root_dimensions[i].as<DimensionNode>()] = op->args[i];
        }

        DimensionPassDownValues(s, vardim_op, current_dim_dom_map, &state, true);

        Array<PrimExpr> args;
        for (auto dim : s->dim_relation_graph->leaf_dimensions) {
          CHECK(state.count(dim.as<DimensionNode>()))
              << "[REPL] Dim " << dim << " " << state[dim.as<DimensionNode>()] << std::endl;
          args.push_back(state[dim.as<DimensionNode>()]);
        }

        ///////////////////////////////////////////////// TESTESTEST
        // if (root_layouts.size() > 0) {
        // lower_tensor_access(s->op->name, s, args, state, root_layouts[op->value_index],
        // old_op->output_layout(op->value_index));
        // }
        ///////////////////////////////////////////////// TESTESTEST

        Array<Range> call_realize_bounds;
        {
          if (print) {
            std::cout << "[RIG]   Generating realize bounds for the access" << std::endl;
          }
          if (auto compute_op = old_op.as<BaseComputeOpNode>()) {
            Region realize_bounds = compute_op->GetRealizeBounds(s, dom_map);

            std::unordered_map<const VarNode*, PrimExpr> vsub;
            CHECK_EQ(compute_op->axis.size(), op->args.size()) << GetRef<PrimExpr>(op) << " " << s;
            for (size_t i = 0; i < compute_op->axis.size(); ++i) {
              auto iv = compute_op->axis[i];
              vsub[iv->var.operator->()] = op->args[i];
            }
            VarReplacer replacer(vsub);
            Region replaced_realize_bounds;
            for (auto r : realize_bounds) {
              auto replaced = replacer.replace(r);
              call_realize_bounds.push_back(replaced);
              if (print) {
                std::cout << "[RIG]    Bound " << r << std::endl;
                std::cout << "[RIG]          " << replaced << std::endl;
              }
            }
          }
        }

        return CallNode::make(op->dtype, this->repl_op->name, args, op->call_type, op->arg_dims,
                              this->repl_op, op->value_index, call_realize_bounds);
      } else if (op->func.as<UninterpFunNode>()) {
        UninterpFun old_fun = Downcast<UninterpFun>(op->func);
        UninterpFun new_fun = replaceUf(old_fun);

        bool changed = !new_fun.same_as(old_fun);
        Array<PrimExpr> new_args;
        for (const auto& arg : op->args) {
          PrimExpr new_arg = this->VisitExpr(arg);
          if (!arg.same_as(new_arg)) changed = true;
          new_args.push_back(new_arg);
        }

        Array<Range> new_custom_realize_bounds;
        for (const auto& bound : op->custom_realize_bounds) {
          auto new_bound = Range::make_by_min_extent(this->VisitExpr(bound->min),
                                                     this->VisitExpr(bound->extent));
          if (!bound.same_as(new_bound)) changed = true;
          new_custom_realize_bounds.push_back(new_bound);
        }

        if (changed)
          return CallNode::make(op->dtype, op->name, new_args, op->call_type, op->arg_dims, new_fun,
                                op->value_index, new_custom_realize_bounds);
        else
          return GetRef<PrimExpr>(op);
      } else {
        return ExprMutator::VisitExpr_(op);
      }
    }

   public:
    UninterpFun replaceUf(UninterpFun orig_) {
      // bool print = (vardim_op->name == "css_update");
      UninterpFun old_orig;
      Array<Dimension> old_new_param_dims;
      Array<Var> old_new_params;

      std::swap(this->orig, old_orig);
      std::swap(this->new_param_dims, old_new_param_dims);
      std::swap(this->new_params, old_new_params);

      this->orig = orig_;
      this->new_param_dims = Array<Dimension>();
      this->new_params = Array<Var>();

      // if (print) std::cout << "[UFREPL]  " << orig->body << std::endl;
      PrimExpr body = orig->body.defined() ? this->VisitExpr(orig->body) : orig->body;
      // if (print) std::cout << "[UFREPL]  " << body << std::endl;
      UninterpFun ret = orig;
      if (!body.same_as(orig->body)) {
        Array<Var> parameters = Array<Var>(orig->parameters);
        Array<Dimension> dimensions = Array<Dimension>(orig->dimensions);
        for (size_t i = 0; i < new_params.size(); ++i) {
          parameters.push_back(new_params[i]);
          dimensions.push_back(new_param_dims[i]);
        }
        ret = UninterpFunNode::make(orig->fname + ".r", orig->range, dimensions, parameters, body,
                                    orig->type);
      }
      std::swap(this->orig, old_orig);
      std::swap(this->new_param_dims, old_new_param_dims);
      std::swap(this->new_params, old_new_params);
      // std::cout << "[UFREPLRET]  " << ret->body << std::endl;
      return ret;
    }

    Replacer(Stage s_, Operation old_op_, Operation repl_op_, const BaseVarDimOpNode* vardim_op_,
             const Map<IterVar, Range>& dom_map_, Array<Modes> root_layouts_)
        : s(s_),
          old_op(old_op_),
          repl_op(repl_op_),
          vardim_op(vardim_op_),
          root_layouts(root_layouts_) {
      for (auto di : vardim_op->GetAllDimensions()) {
        if (dom_map_.count(di->iv)) {
          current_dim_dom_map[di->dim.as<DimensionNode>()] = dom_map_.at(di->iv);
        }
      }
      for (auto it : dom_map_) {
        dom_map[it.first] = it.second;
      }
    }

    Stage s;
    Operation old_op;
    Operation repl_op;
    const BaseVarDimOpNode* vardim_op;
    Array<Modes> root_layouts;
    std::unordered_map<IterVar, Range> dom_map;

    Array<Dimension> orig_idx_dims;
    std::unordered_map<const DimensionNode*, Range> current_dim_dom_map;

    UninterpFun orig = NullValue<UninterpFun>();
    Array<Dimension> new_param_dims;
    Array<Var> new_params;
  };

  // std::cout << "[RIG] Replacing accesses in " << reader << std::endl;

  if (auto compute_op = reader.as<ComputeOpNode>()) {
    auto new_op = make_object<ComputeOpNode>(*compute_op);
    bool print = false;  //(compute_op->name == "ii_s_h2h.ila");
    if (print) std::cout << "[RI] Replacing in " << compute_op->name << std::endl;
    bool changed = false;
    Replacer replacer(s, old_op, repl_op, compute_op, dom_map, root_layouts);

    Array<PrimExpr> arr;
    if (compute_op->body[0]->IsInstance<tir::ReduceNode>()) {
      // Specially handle reduce so the replaced op
      // still share all the components
      PrimExpr new_reduce = replacer(compute_op->body[0]);
      if (!new_reduce.same_as(compute_op->body[0])) {
        const tir::ReduceNode* r = new_reduce.as<tir::ReduceNode>();
        for (size_t k = 0; k < compute_op->body.size(); ++k) {
          auto n = make_object<tir::ReduceNode>(*r);
          n->value_index = static_cast<int>(k);
          n->dtype = r->source[k].dtype();
          arr.push_back(PrimExpr(n));
        }
      } else {
        arr = compute_op->body;
      }
    } else {
      for (auto e : compute_op->body) {
        if (print) std::cout << "[RI]    Body " << e << std::endl;
        PrimExpr new_expr = replacer(e);
        // std::cout << "[RI]  Body replaced to " << e << " " << new_expr << std::endl;
        arr.push_back(new_expr);
      }
    }

    Array<PrimExpr> pred_arr;
    for (auto e : compute_op->pred) {
      PrimExpr new_expr = replacer(e);
      if (print) std::cout << "[RI]  Replaced to " << new_expr << std::endl;
      pred_arr.push_back(new_expr);
    }

    Array<DimInfo> new_dim_infos;
    for (const auto di : new_op->all_dimensions) {
      CHECK(!di->dim->isFunDim());
      IterVar iv = di->iv;
      PrimExpr old_extent = iv->dom->extent;
      PrimExpr new_extent = replacer(old_extent);
      if (!new_extent.same_as(old_extent)) {
        if (print)
          std::cout << "[REPL]  Extent " << UninterpFun::InlineUninterpFunCalls(old_extent) << " "
                    << UninterpFun::InlineUninterpFunCalls(new_extent) << std::endl;
        const_cast<RangeNode*>(iv->dom.as<RangeNode>())->extent = new_extent;
        changed = true;
      }
      // if (print) std::cout << "[REPL] " << di->iv << std::endl;
      new_dim_infos.push_back(DimInfoNode::make(di->dim, di->iv));
    }
    new_op->set_all_dimensions(new_dim_infos);

    if (!arr.same_as(compute_op->body)) {
      new_op->body = arr;
      changed = true;
    }

    if (!pred_arr.same_as(compute_op->pred)) {
      new_op->pred = pred_arr;
      changed = true;
    }

    if (changed) {
      new_op->RefreshDimVarMappings();
      new_op->set_realize_bounds(compute_op->realize_bounds, compute_op->who_set_realize_bounds);
      if (print) std::cout << "[REPL] Returning new" << std::endl;
      return Operation(new_op);
    } else
      return reader;
  } else if (auto scan_op = reader.as<ScanOpNode>()) {
    // std::cout << "[REPL] OP " << reader << std::endl;
    auto new_op = make_object<ScanOpNode>(*scan_op);
    bool changed = false;
    Replacer replacer(s, old_op, repl_op, scan_op, dom_map, root_layouts);

    std::vector<std::unordered_map<const DimensionNode*, DimVarEntry>> new_dim2var_maps;
    for (auto& dim2var_map : new_op->dim2var_maps) {
      std::unordered_map<const DimensionNode*, DimVarEntry> new_dim2var_map;
      for (auto& it : dim2var_map) {
        IterVar iv = it.second.iv;
        PrimExpr old_extent = iv->dom->extent;
        PrimExpr new_extent = replacer(old_extent);
        if (!new_extent.same_as(old_extent)) {
          // std::cout << "[REPL]   Extent " << old_extent << " " << new_extent << " " << it.first
          //           << std::endl;
          const_cast<RangeNode*>(iv->dom.as<RangeNode>())->extent = new_extent;
          changed = true;
        }

        CHECK(!it.first->isFunDim());
        new_dim2var_map[it.first] = {it.second.dim, it.second.iv};
      }
      new_dim2var_maps.push_back(new_dim2var_map);
    }
    new_op->dim2var_maps = new_dim2var_maps;

    if (changed) {
      Operation op = Operation(new_op);
      for (auto t : op->InputTensors()) {
        // std::cout << "[REPL]  Input tensor " << t << std::endl;
      }
      return op;
    } else
      return reader;
  } else if (auto sk_op = reader.as<SingleKernelEnvelopeOpNode>()) {
    // std::cout << "[REPL] OP " << reader << std::endl;
    auto new_op = make_object<SingleKernelEnvelopeOpNode>(*sk_op);
    bool changed = false;
    Replacer replacer(s, old_op, repl_op, sk_op, dom_map, root_layouts);

    std::vector<std::unordered_map<const DimensionNode*, DimVarEntry>> new_dim2var_maps;
    for (auto& dim2var_map : new_op->dim2var_maps) {
      std::unordered_map<const DimensionNode*, DimVarEntry> new_dim2var_map;
      for (auto& it : dim2var_map) {
        IterVar iv = it.second.iv;
        PrimExpr old_extent = iv->dom->extent;
        PrimExpr new_extent = replacer(old_extent);
        if (!new_extent.same_as(old_extent)) {
          // std::cout << "[REPL]   Extent " << iv << " "
          //           << UninterpFun::InlineUninterpFunCalls(old_extent) << " "
          //           << UninterpFun::InlineUninterpFunCalls(new_extent) << " " << it.first
          //           << std::endl;
          const_cast<RangeNode*>(iv->dom.as<RangeNode>())->extent = new_extent;
          changed = true;
        }

        CHECK(!it.first->isFunDim());
        new_dim2var_map[it.first] = {it.second.dim, it.second.iv};
      }
      new_dim2var_maps.push_back(new_dim2var_map);
    }
    new_op->dim2var_maps = new_dim2var_maps;

    if (changed) {
      Operation op = Operation(new_op);
      for (auto t : op->InputTensorsWithUnemitted()) {
        // std::cout << "[REPL]  Input tensor " << t << std::endl;
      }
      return op;
    } else
      return reader;
  } else if (auto conditional_op = reader.as<ConditionalOpNode>()) {
    // std::cout << "[REPL] OP " << reader << std::endl;
    auto new_op = make_object<ConditionalOpNode>(*conditional_op);
    bool changed = false;
    Replacer replacer(s, old_op, repl_op, conditional_op, dom_map, root_layouts);

    std::vector<std::unordered_map<const DimensionNode*, DimVarEntry>> new_dim2var_maps;
    for (auto& dim2var_map : new_op->dim2var_maps) {
      std::unordered_map<const DimensionNode*, DimVarEntry> new_dim2var_map;
      for (auto& it : dim2var_map) {
        IterVar iv = it.second.iv;
        PrimExpr old_extent = iv->dom->extent;
        PrimExpr new_extent = replacer(old_extent);
        if (!new_extent.same_as(old_extent)) {
          // std::cout << "[REPL]   Extent " << old_extent << " " << new_extent << " " << it.first
          //           << std::endl;
          const_cast<RangeNode*>(iv->dom.as<RangeNode>())->extent = new_extent;
          changed = true;
        }

        CHECK(!it.first->isFunDim());
        new_dim2var_map[it.first] = {it.second.dim, it.second.iv};
      }
      new_dim2var_maps.push_back(new_dim2var_map);
    }
    new_op->dim2var_maps = new_dim2var_maps;

    PrimExpr new_condition = replacer(conditional_op->condition);
    if (!new_condition.same_as(conditional_op->condition)) {
      changed = true;
      new_op->condition = new_condition;
    }

    if (changed) {
      Operation op = Operation(new_op);
      for (auto t : op->InputTensors()) {
        // std::cout << "[REPL]  Input tensor " << t << std::endl;
      }
      return op;
    } else
      return reader;
  } else {
    if (!reader.as<ExternOpNode>())
      CHECK(false) << "Only scan and compute readers supported " << reader;
    return reader;
  }
}

}  // namespace te
}  // namespace tvm
