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
  bool print = false;//(this->tensor->op->name == "left");
  if (!op->func.defined()) ExprVisitor::VisitExpr_(op);
  if (op->func.as<OperationNode>()) {
    Tensor t = Downcast<Operation>(op->func).output(op->value_index);
    if (t->op.defined() && t == this->tensor) {
      if (print)
        std::cout << "[AP] Access found " << GetRef<PrimExpr>(op) << " "
                  << original_index_dimensions.size() << std::endl;
      AccessPattern* ap = new AccessPattern();
      for (size_t i = 0; i < original_index_dimensions.size(); ++i) {
        if (print) std::cout << "[AP]   Dim " << original_index_dimensions[i] << std::endl;
        if (original_index_dimensions[i]->isFunDim()) {
          PrimExpr arg = op->args[i];
          if (arg.as<VarNode>()) {
            auto var = Downcast<Var>(arg);
            if (print) std::cout << "[AP]     looking for var " << var << std::endl;
            ap->idx_dim_args.Set(original_index_dimensions[i], GetDimForVar(var));
          } else {
            if (print) std::cout << "[AP]     Non var arg " << arg << std::endl;
          }
        }
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
        if (di->dim->isFunDim()) {
          UninterpFun ufun = di->ufun;
          Map<Var, Dimension> ufun_var2dim_map;
          for (size_t j = 0; j < ufun->arity(); ++j) {
            ufun_var2dim_map.Set(ufun->parameters[j], ufun->dimensions[j]);
          }
          exprCollector.collect(ufun.as<UninterpFunNode>(), ufun_var2dim_map, 0);
        } else {
          // std::cout << "[AP]   Extent " << di->iv->dom->extent << std::endl;
          exprCollector.collect(di->iv->dom->extent, op_var2dim_map, 0);
        }
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
          if (it.first->isFunDim()) {
            UninterpFun ufun = it.second.value_expr;
            // std::cout << "[AP]   Dim " << it.first->name << std::endl;

            Map<Var, Dimension> ufun_var2dim_map = Map<Var, Dimension>(var2dim_map);
            for (size_t i = 0; i < ufun->dimensions.size(); ++i) {
              ufun_var2dim_map.Set(ufun->parameters[i], ufun->dimensions[i]);
            }

            exprCollector.collect(ufun.as<UninterpFunNode>(), ufun_var2dim_map, i);
          }
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
          if (it.first->isFunDim()) {
            UninterpFun ufun = it.second.value_expr;
            // std::cout << "[AP]   Dim " << it.first->name << std::endl;

            Map<Var, Dimension> ufun_var2dim_map = Map<Var, Dimension>(var2dim_map);
            for (size_t i = 0; i < ufun->dimensions.size(); ++i) {
              ufun_var2dim_map.Set(ufun->parameters[i], ufun->dimensions[i]);
            }

            exprCollector.collect(ufun.as<UninterpFunNode>(), ufun_var2dim_map, i);
          }
          exprCollector.collect(it.second.iv->dom->extent, var2dim_map, i);
        }
      }
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
        // std::cout << "[RI]    Found call " << GetRef<PrimExpr>(op) << " " << op << " " << pattern
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
        PrimExpr new_call = CallNode::make(op->dtype, this->cache->op->name, args, op->call_type,
                                           this->cache->op, this->cache->value_index);
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
      return UninterpFunNode::make(orig->fname + ".r", orig->range, dimensions, parameters, body);
    }

    UninterpFun orig;
    Array<Dimension> new_param_dims;
    Array<Var> new_params;
  };

  class Replacer : public ExprMutator {
    PrimExpr VisitExpr_(const CallNode* op) override {
      // bool print = (vardim_op->name == "css_update");
      if (this->patterns_map->find(op) != this->patterns_map->end()) {
        // std::cout << "[RI] Found call " << GetRef<PrimExpr>(op) << std::endl;
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
        PrimExpr new_call = CallNode::make(op->dtype, this->cache->op->name, args, op->call_type,
                                           this->cache->op, this->cache->value_index);
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

        if (changed)
          return CallNode::make(op->dtype, op->name, new_args, op->call_type,
                                op->argument_dimensions, new_fun, op->value_index);
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
        ret = UninterpFunNode::make(orig->fname + ".r", orig->range, dimensions, parameters, body);
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
    bool print = false;  //(compute_op->name == "h_gate");
    if (print) std::cout << "[RI] Replacing in " << compute_op->name << std::endl;
    bool changed = false;
    ExprReplacer expr_replacer(compute_op, patterns_map, cache, cache_idx_dims, orig_idx_dims,
                               add_variant_dimension);
    Array<PrimExpr> arr;
    if (compute_op->body[0]->IsInstance<tir::ReduceNode>()) {
      // Specially handle reduce so the replaced op
      // still share all the components
      PrimExpr new_reduce = expr_replacer(compute_op->body[0]);
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
        if (print) std::cout << "[RI]  Replaced to " << new_expr << std::endl;
        arr.push_back(new_expr);
      }
    }

    UFReplacer uf_replacer(patterns_map, cache, cache_idx_dims, orig_idx_dims,
                           add_variant_dimension);
    Replacer new_replacer(patterns_map, cache, cache_idx_dims, orig_idx_dims, add_variant_dimension,
                          compute_op);

    Array<DimInfo> new_dim_infos;
    for (const auto di : new_op->all_dimensions) {
      if (di->dim->isFunDim()) {
        UninterpFun old_fun = di->ufun;
        UninterpFun new_fun = uf_replacer.replace(old_fun);
        if (!new_fun.same_as(old_fun)) {
          if (print)
            std::cout << "[REPL]  UF " << old_fun->body << " " << new_fun->body << std::endl;
          changed = true;
        }
        new_dim_infos.push_back(DimInfoNode::make(di->dim, di->iv, new_fun));
      } else {
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
        new_dim_infos.push_back(DimInfoNode::make(di->dim, di->iv, di->ufun));
      }
    }
    new_op->set_all_dimensions(new_dim_infos);

    if (!arr.same_as(compute_op->body)) {
      new_op->body = arr;
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

        if (it.first->isFunDim()) {
          UninterpFun old_fun = it.second.value_expr;
          UninterpFun new_fun = uf_replacer.replace(old_fun);
          // std::cout << "[REPL]    ufun " << old_fun->body << " " << new_fun->body << std::endl;
          if (!new_fun.same_as(old_fun)) {
            new_dim2var_map[it.first] = {it.second.dim, it.second.iv, new_fun};
            changed = true;
          } else {
            new_dim2var_map[it.first] = {it.second.dim, it.second.iv, old_fun};
          }
        } else {
          new_dim2var_map[it.first] = {it.second.dim, it.second.iv, it.second.value_expr};
        }
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
          // std::cout << "[REPL]   Extent " << old_extent << " " << new_extent << " " << it.first
          // << std::endl;
          const_cast<RangeNode*>(iv->dom.as<RangeNode>())->extent = new_extent;
          changed = true;
        }

        if (it.first->isFunDim()) {
          UninterpFun old_fun = it.second.value_expr;
          UninterpFun new_fun = uf_replacer.replace(old_fun);
          // std::cout << "[REPL]    ufun " << old_fun->body << " " << new_fun->body << std::endl;
          if (!new_fun.same_as(old_fun)) {
            new_dim2var_map[it.first] = {it.second.dim, it.second.iv, new_fun};
            changed = true;
          } else {
            new_dim2var_map[it.first] = {it.second.dim, it.second.iv, old_fun};
          }
        } else {
          new_dim2var_map[it.first] = {it.second.dim, it.second.iv, it.second.value_expr};
        }
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
  } else {
    CHECK(false) << "Only scan and compute readers supported";
    return reader;
  }
}

}  // namespace te
}  // namespace tvm
