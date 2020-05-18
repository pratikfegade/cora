#include "tensor_layout_utils.h"

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

void AccessPatternCollector::ExprAccessPatternCollector::VisitExpr_(const CallNode* op) {
  if (op->func.defined()) {
    Tensor t = Downcast<Operation>(op->func).output(op->value_index);
    // std::cout << "CHECKING " << GetRef<PrimExpr>(op) << " " << (t == this->tensor) << " "
    //           << op->func << " " << this->tensor->op << std::endl;
    if (t->op.defined() && t == this->tensor) {
      AccessPattern* ap = new AccessPattern();

      for (size_t i = 0; i < original_index_dimensions.size(); ++i) {
        if (original_index_dimensions[i]->type == DimensionNode::kFunDim) {
          PrimExpr arg = op->args[i];
          if (arg.as<VarNode>()) {
            auto var = Downcast<Var>(arg);
            ap->idx_dim_args.Set(original_index_dimensions[i], var2dim_map.at(var));
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
  }
  ExprVisitor::VisitExpr_(op);
}

void AccessPatternCollector::ExprAccessPatternCollector::collect(const UninterpFunNode* ufun,
                                                                 Map<Var, Dimension> var2dim_map_,
                                                                 int reader_val_idx_) {
  this->var2dim_map = var2dim_map_;
  this->ufun = ufun;
  this->reader_val_idx = reader_val_idx_;
  this->operator()(ufun->body);
}

void AccessPatternCollector::ExprAccessPatternCollector::collect(PrimExpr expr,
                                                                 Map<Var, Dimension> var2dim_map_,
                                                                 int reader_val_idx_) {
  this->var2dim_map = var2dim_map_;
  this->ufun = nullptr;
  this->reader_val_idx = reader_val_idx_;
  this->operator()(expr);
}

void AccessPatternCollector::collect() {
  for (auto reader : this->readers) {
    if (auto reader_op = reader.as<ComputeOpNode>()) {
      ExprAccessPatternCollector exprCollector(this->tensor, original_index_dimensions,
                                               &(this->access_patterns),
                                               &(this->access_to_pattern_map), reader_op);
      {
        Map<Var, Dimension> var2dim_map;
        for (const auto& it : reader_op->var2dim_map) {
          var2dim_map.Set(GetRef<Var>(it.first), GetRef<Dimension>(it.second));
        }
        for (const auto& body_expr : reader_op->body) {
          exprCollector.collect(body_expr, var2dim_map, 0);
        }
        for (const auto& iv : reader_op->axis) {
          if (const auto& call = iv->dom->extent.as<CallNode>()) {
            if (const auto& ufun = call->func.as<UninterpFunNode>()) {
              exprCollector.collect(ufun, var2dim_map, 0);
            }
          }
        }
      }
      {
        Map<Var, Dimension> var2dim_map;
        for (auto dim : reader_op->loop_dimensions) {
          var2dim_map.Set(reader_op->GetIterVarFromDim(0, dim)->var, dim);
        }
        for (auto ie : reader_op->index_expressions) {
          exprCollector.collect(ie.as<UninterpFunNode>(), var2dim_map, 0);
        }
      }
    } else if (auto reader_op = reader.as<ScanOpNode>()) {
      ExprAccessPatternCollector exprCollector(this->tensor, original_index_dimensions,
                                               &(this->access_patterns),
                                               &(this->access_to_pattern_map), reader_op);

      Map<Var, Dimension> var2dim_map;
      for (const auto& dim2var_map : reader_op->dim2var_maps) {
        for (const auto& it : dim2var_map) {
          // if (it.first->type == DimensionNode::kFunDim) {
          var2dim_map.Set(it.second.iv->var, GetRef<Dimension>(it.first));
          // }
        }
      }
      for (int i = 0; i < reader_op->num_outputs(); ++i) {
        for (const auto& it : reader_op->dim2var_maps[i]) {
          if (it.first->type == DimensionNode::kFunDim) {
            UninterpFun ufun = it.second.value_expr;
            exprCollector.collect(ufun.as<UninterpFunNode>(), var2dim_map, i);
          }
          if (auto call = it.second.iv->dom->extent.as<CallNode>()) {
            if (call->func.as<UninterpFunNode>()) {
              UninterpFun ufun = Downcast<UninterpFun>(call->func);
              std::cout << "COLLECTING " << ufun->body << std::endl;
              exprCollector.collect(ufun.as<UninterpFunNode>(), var2dim_map, i);
            }
          }
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
          add_variant_dimension(add_variant_dimension_) {}

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
          vardim_op(vardim_op_) {}

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

  if (auto compute_op = reader.as<ComputeOpNode>()) {
    auto new_op = make_object<ComputeOpNode>(*compute_op);
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
        // std::cout << "[RI] Replaced to " << new_expr << std::endl;
        arr.push_back(new_expr);
      }
    }

    UFReplacer uf_replacer(patterns_map, cache, cache_idx_dims, orig_idx_dims,
                           add_variant_dimension);
    Array<UninterpFun> new_index_expressions;
    for (size_t i = 0; i < new_op->index_expressions.size(); ++i) {
      UninterpFun old_fun = new_op->index_expressions[i];
      UninterpFun new_fun = uf_replacer.replace(old_fun);
      if (!new_fun.same_as(old_fun)) {
        // std::cout << "[CRO]   Idx expr " << old_fun->body << " " << new_fun->body << std::endl;
        changed = true;
      }
      new_index_expressions.push_back(new_fun);
    }
    new_op->set_index_expressions(new_index_expressions);

    for (auto iv : new_op->axis) {
      if (auto call = iv->dom->extent.as<CallNode>()) {
        if (call->func.as<UninterpFunNode>()) {
          UninterpFun old_fun = Downcast<UninterpFun>(call->func);
          UninterpFun new_fun = uf_replacer.replace(old_fun);
          if (!new_fun.same_as(old_fun)) {
            const_cast<CallNode*>(call)->func = new_fun;
            changed = true;
          }
        }
      }
    }
    if (!arr.same_as(compute_op->body)) {
      new_op->body = arr;
      changed = true;
    }

    if (changed) {
      new_op->RefreshDimVarMappings();
      return Operation(new_op);
    } else
      return reader;
  } else if (auto scan_op = reader.as<ScanOpNode>()) {
    std::cout << "[REPL] OP " << reader << std::endl;
    auto new_op = make_object<ScanOpNode>(*scan_op);
    bool changed = false;
    UFReplacer uf_replacer(patterns_map, cache, cache_idx_dims, orig_idx_dims,
                           add_variant_dimension);

    for (auto& dim2var_map : new_op->dim2var_maps) {
      for (auto& it : dim2var_map) {
        if (it.first->type == DimensionNode::kFunDim) {
          UninterpFun old_fun = it.second.value_expr;
          UninterpFun new_fun = uf_replacer.replace(old_fun);
          if (!new_fun.same_as(old_fun)) {
            it.second = {it.second.dim, it.second.iv, new_fun};
            changed = true;
          }
        }

        std::cout << "[REPL]   " << it.second.iv->dom->extent << std::endl;
        if (auto call = it.second.iv->dom->extent.as<CallNode>()) {
          if (call->func.as<UninterpFunNode>()) {
            UninterpFun old_fun = Downcast<UninterpFun>(call->func);
            UninterpFun new_fun = uf_replacer.replace(old_fun);
            std::cout << "[REPL]   " << old_fun->fname << " " << old_fun->body << " "
                      << new_fun->fname << " " << new_fun->body << std::endl;
            if (!new_fun.same_as(old_fun)) {
              const_cast<CallNode*>(call)->func = new_fun;
              changed = true;
            }
          }
        }
      }
    }

    if (changed)
      return Operation(new_op);
    else
      return reader;
  } else {
    CHECK(false) << "Only scan and compute readers supported";
    return reader;
  }
}

}  // namespace te
}  // namespace tvm
