#include <tvm/te/schedule.h>
#include <tvm/te/operation.h>
#include <tvm/tir/expr_equality.h>
#include <tvm/ir/attrs.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/ir_pass.h>
#include <unordered_set>
#include "message_passing.h"
#include "../../tir/pass/ir_util.h"
#include "../../arith/compute_expr.h"
#include "../../tir/ir/var_replacer.h"
#include "schedule_utils.h"

namespace tvm {
  namespace te {
    class AccessPattern {
    public:
      // For each dimension of the tensor indexed by an Fdim, what
      // dimension in the reader access is used to index into the
      // tensor?
      Map<Dimension, Dimension> idx_dim_args;
      const CallNode* original_access;
      const ComputeOpNode* reader_op;
      const UninterpFunNode* ufun;
      int idx;

      class Hasher {
      public:
	size_t operator()(const AccessPattern* pattern) const {
	  AttrsHash hasher;
	  using std::hash;
	  size_t h = 0;
	  for (auto it: pattern->idx_dim_args) {
	    Dimension d = it.first;
	    Dimension idx = it.second;
	    h += hasher(d) + hasher(idx);
	  }
	  return h;
	}
      };
      class Equality {
      public:
	bool operator()(const AccessPattern* p1, const AccessPattern* p2) const {
	  AttrsEqual equals;
	  for (auto it1: p1->idx_dim_args) {
	    Dimension d1 = it1.first;
	    Dimension idx1 = it1.second;
	    if (p2->idx_dim_args.count(d1)) {
	      Dimension idx2 = p2->idx_dim_args.at(d1);
	      if (idx1 != idx2) {
		return false;
	      }
	    }
	    else {
	      return false;
	    }
	  }
	  return true;
	}
      };
    };

    using PatternsSet = std::unordered_set<AccessPattern*, AccessPattern::Hasher, AccessPattern::Equality>;
    using AccessToPatternMap = std::unordered_map<const CallNode*, AccessPattern*>;
    using PatternsVec = std::vector<AccessPattern*>;

    class AccessPatternCollector {
      class ExprAccessPatternCollector : public ExprVisitor {
	PrimExpr ExpandToLoopVars(PrimExpr expr, const ComputeOpNode* op) {
	  class Expander: public ExprMutator {
	    PrimExpr VisitExpr_(const VarNode* op) override {
	      for (size_t i = 0; i < reader_op->index_variables.size(); ++i) {
		if (op == reader_op->index_variables[i]->var.get()) {
		  Array<PrimExpr> loop_vars;
		  for (auto iv: reader_op->axis) {
		    loop_vars.push_back(iv->var);
		  }
		  return reader_op->index_expressions[i]->
		    substitute(loop_vars, reader_op->loop_dimensions);
		}
	      }
	      return ExprMutator::VisitExpr_(op);
	    }

	  public:
	    Expander(const ComputeOpNode* reader_op_) : reader_op(reader_op_) {}
	    const ComputeOpNode* reader_op;
	  };
	  return Expander(reader_op)(expr);
	}

	void VisitExpr_(const CallNode* op) override {
	  if (op->func.defined()) {
	    Tensor t = Downcast<Operation>(op->func).output(op->value_index);
	    if (t->op.defined() && t == this->tensor) {
	      AccessPattern* ap = new AccessPattern();

	      for (size_t i = 0; i < original_index_dimensions.size(); ++i) {
		if (original_index_dimensions[i]->type == DimensionNode::kFunDim) {
		  PrimExpr arg = op->args[i];
		  if (auto var = arg.as<VarNode>()) {
		    ap->idx_dim_args.Set(original_index_dimensions[i],
					 GetRef<Dimension>(reader_op->var2dim_map.at(var)));
		  }
		}
	      }



	      // for (size_t i = 0; i < op->args.size(); ++i) {
	      // 	auto expanded = this->ExpandToLoopVars(op->args[i], reader_op);
	      // 	ap->idx_dim_args.Set(tensor_index_dims[i], expanded);
	      // }
	      ap->original_access = op;
	      ap->reader_op = reader_op;
	      ap->ufun = this->ufun;
	      if (this->access_patterns->insert(ap).second) {
		// std::cout << "[CRO] New pattern " << GetRef<PrimExpr>(op) << " " << op << " in " << reader_op->name << std::endl;
		// for (auto iv: reader_op->index_variables) {
		//   std::cout << "[CRO]  IV " << iv << std::endl;
		// }
		// for (auto iv: reader_op->axis) {
		//   std::cout << "[CRO]  LV " << iv << std::endl;
		// }
	      }
	      (*this->access_to_pattern_map)[op] = ap;
	    }
	  }
	  ExprVisitor::VisitExpr_(op);
	}

      public:
	ExprAccessPatternCollector(const Tensor& tensor_, Array<Dimension> original_index_dimensions_,
				   PatternsSet* access_patterns_,
				   AccessToPatternMap* access_to_pattern_map_, const ComputeOpNode* reader_op_) :
	  tensor(tensor_), original_index_dimensions(original_index_dimensions_), access_patterns(access_patterns_),
	  access_to_pattern_map(access_to_pattern_map_), reader_op(reader_op_) {
	  if (auto op = tensor->op.as<ComputeOpNode>()) {
	    this->tensor_index_dims = op->root_index_dimensions;
	  }
	  else if (auto op = tensor->op.as<PlaceholderOpNode>()) {
	    this->tensor_index_dims = op->self_index_dimensions;
	  }
	  else {
	    CHECK(false) << "Cannot only cache Compute and Plcceholder operations";
	  }
	}

	void collect(PrimExpr expr, const UninterpFunNode* ufun) {
	  this->ufun = ufun;
	  this->operator()(expr);
	}

	void collect(PrimExpr expr) {
	  this->ufun = nullptr;
	  this->operator()(expr);
	}

	const Tensor& tensor;
	Array<Dimension> original_index_dimensions;
	PatternsSet* access_patterns;
	AccessToPatternMap* access_to_pattern_map;
	const ComputeOpNode* reader_op;
	Array<Dimension> tensor_index_dims;
	const UninterpFunNode* ufun;
      };

      void collectAccesPatterns() {
	for (auto reader: this->readers) {
	  if (auto reader_op = reader.as<ComputeOpNode>()) {
	    ExprAccessPatternCollector exprCollector(this->tensor, original_index_dimensions, &(this->access_patterns),
						     &(this->access_to_pattern_map), reader_op);
	    for (auto body_expr: reader_op->body) {
	      exprCollector.collect(body_expr);
	    }
	    for (auto ie: reader_op->index_expressions) {
	      exprCollector.collect(ie->body, ie.as<UninterpFunNode>());
	      // Array<PrimExpr> args;
	      // for (auto iv: reader_op->axis) {
	      // args.push_back(iv->var);
	      // }
	      // exprCollector(UninterpFun::InlineUninterpFunCalls(
	      // CallNode::make(DataType::Int(32), ie->fname, args,
	      // CallNode::PureExtern, reader_op->loop_dimensions, ie, 0)));
	    }
	    for (auto iv: reader_op->axis) {
	      if (auto call = iv->dom->extent.as<CallNode>()) {
		if (auto ufun = call->func.as<UninterpFunNode>()) {
		  exprCollector.collect(ufun->body, ufun);
		}
	      }
	      // std::cout << "[CRO] AXIS " << iv->dom->extent << std::endl;
	      // exprCollector(UninterpFun::InlineUninterpFunCalls(iv->dom->extent));
	    }
	  }
	  else {
	    CHECK(false) <<
	      "Opaque caching is not yet implemented for reader op " << reader;
	  }
	}
      }

    public:
      void collect() {
	collectAccesPatterns();
      }

      AccessPatternCollector(const Tensor& tensor_, Array<Dimension> original_index_dimensions_,
			     const Array<Operation>& readers_) :
	tensor(tensor_), original_index_dimensions(original_index_dimensions_), readers(readers_) {}

      const Tensor& tensor;
      Array<Dimension> original_index_dimensions;
      const Array<Operation>& readers;
      PatternsSet access_patterns;
      AccessToPatternMap access_to_pattern_map;
    };

    class TranslateVarsCrossStages: public ExprMutator {
      PrimExpr VisitExpr_(const VarNode* op) override {

	Dimension var_dim;
	for (size_t i = 0; i < reader_op->index_variables.size(); ++i) {
	  // std::cout << "[CRO] Dim " << reader_op->index_dimensions[i]->name << std::endl;
	  if (op == reader_op->index_variables[i]->var.get()) {
	    var_dim = reader_op->index_dimensions[i];
	  }
	}

	for (size_t i = 0; i < reader_op->axis.size(); ++i) {
	  if (op == reader_op->axis[i]->var.get()) {
	    var_dim = reader_op->loop_dimensions[i];
	  }
	}

	// std::cout << "[CRO] Dim found " << var_dim << std::endl;

	for (size_t i = 0; i < index_dimensions.size(); ++i) {
	  if (var_dim == index_dimensions[i]) {
	    return index_variables[i]->var;
	  }
	}

	for (size_t i = 0; i < loop_dimensions.size(); ++i) {
	  if (var_dim == loop_dimensions[i]) {
	    return loop_variables[i]->var;
	  }
	}
	return ExprMutator::VisitExpr_(op);
      }

    public:
      TranslateVarsCrossStages(const CallNode* op_, const ComputeOpNode* reader_op_, Array<IterVar>& index_variables_,
			       Array<IterVar>& loop_variables_, Array<Dimension>& index_dimensions_,
			       Array<Dimension>& loop_dimensions_) :
	op(op_), reader_op(reader_op_), index_variables(index_variables_),
	loop_variables(loop_variables_), index_dimensions(index_dimensions_),
	loop_dimensions(loop_dimensions_){}

      const CallNode* op;
      const ComputeOpNode* reader_op;
      Array<IterVar> index_variables;
      Array<IterVar> loop_variables;
      Array<Dimension> index_dimensions;
      Array<Dimension> loop_dimensions;
    };

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

    PrimExpr CacheBodyBuilder(Tensor tensor, Array<Dimension>& original_index_dimensions,
			      const PatternsVec& patterns_vec, Array<IterVar>& index_variables,
			      Array<IterVar>& loop_variables, Array<Dimension>& index_dimensions,
			      Array<Dimension>& loop_dimensions) {
      const Var variant_loop_var = loop_variables[loop_variables.size() - 1]->var;

      PrimExpr body = PrimExpr(0);
      for (size_t i = 0; i < patterns_vec.size(); ++i) {
	AccessPattern* pattern = patterns_vec[i];
	PrimExpr expr;
	// if (pattern->ufun != nullptr) {
	//   auto tmp = UninterpFunNode::make(pattern->ufun->fname, pattern->ufun->range, pattern->ufun->dimensions,
	// 				   pattern->ufun->parameters, GetRef<PrimExpr>(pattern->original_access));
	//   Array<PrimExpr> args;
	//   Array<Dimension> arg_dims;
	//   for (size_t i = 0; i < loop_dimensions.size(); ++i) {
	//     args.push_back(loop_variables[i]);
	//     arg_dims.push_back(loop_dimensions[i]);
	//   }
	//   for (size_t i = 0; i < index_dimensions.size(); ++i) {
	//     args.push_back(index_variables[i]);
	//     arg_dims.push_back(index_dimensions[i]);
	//   }
	//   expr = tmp->substitute(args,arg_dims);
	// }
	// else {
	  Array<PrimExpr> args;
	  for (size_t i = 0; i < original_index_dimensions.size(); ++i) {
	    if (original_index_dimensions[i]->type == DimensionNode::kFunDim) {
	      Dimension arg_dim = pattern->idx_dim_args.at(original_index_dimensions[i]);
	      index_dimensions.push_back(arg_dim);
	      auto reader_iv = pattern->reader_op->GetIterVarFromDim(arg_dim);
	      auto iv = IterVarNode::make(reader_iv->dom, reader_iv->var.copy_with_suffix(""),
					  reader_iv->iter_type, reader_iv->thread_tag);
	      index_variables.push_back(iv);
	      args.push_back(iv->var);
	    }
	    else {
	      args.push_back(GetIterVarFromDim(original_index_dimensions[i], index_variables, loop_variables,
					       index_dimensions, loop_dimensions));
	    }
	    expr = CallNode::make(DataType::Int(32), tensor->op->name, args, CallNode::Halide, tensor->op, 0);
	  // }


	  expr = TranslateVarsCrossStages(pattern->original_access, pattern->reader_op, index_variables,
					  loop_variables, index_dimensions, loop_dimensions)
	    (GetRef<PrimExpr>(pattern->original_access));
	}

	body = if_then_else(variant_loop_var == static_cast<int>(i), expr, body);
      }
      return body;
    }

    Operation ReplaceInputs(Operation reader, const AccessToPatternMap* patterns_map,
			    Tensor cache, Array<Dimension> cache_idx_dims, Array<Dimension> orig_idx_dims) {
      class ExprReplacer: public ExprMutator {
	PrimExpr VisitExpr_(const CallNode* op) override {
	  if (this->patterns_map->find(op) != this->patterns_map->end()) {
	    auto pattern = this->patterns_map->find(op)->second;
	    Array<PrimExpr> args;
	    // Skip the last dimension as that's the variant dimension
	    // we handle after the loop
	    for (size_t i = 0; i < cache_idx_dims.size() - 1; ++i) {
	      auto dim = cache_idx_dims[i];
	      if (!orig_idx_dims.Contains(dim)) {
		// This is newly added dimensions, corresponding to an
		// index of the original tensor. For this, we need to
		// index by the IV corresponding to this dimension.
		args.push_back(compute_op->GetIterVarFromDim(dim)->var);
	      }
	      else {
		// Here we leave the argument intact, for the case
		// where the dimension is left unmodified by the
		// transform.
		args.push_back(op->args[i]);
	      }
	    }
	    args.push_back(pattern->idx);
	    PrimExpr new_call = CallNode::make(op->dtype, this->cache->op->name, args, op->call_type,
					       this->cache->op, this->cache->value_index);
	    std::cout << "[CRO]  Replacing " << GetRef<PrimExpr>(op) << " " << new_call << std::endl;
	    return new_call;
	  }
	  else return ExprMutator::VisitExpr_(op);
	}

      public:
	ExprReplacer(const ComputeOpNode* compute_op_, const AccessToPatternMap* patterns_map_,
		 Tensor cache_, Array<Dimension> cache_idx_dims_, Array<Dimension> orig_idx_dims_) :
	  compute_op(compute_op_), patterns_map(patterns_map_), cache(cache_), cache_idx_dims(cache_idx_dims_) {}

	const ComputeOpNode* compute_op;
	const AccessToPatternMap* patterns_map;
	Tensor cache;
	Array<Dimension> cache_idx_dims;
	Array<Dimension> orig_idx_dims;
      };

      class UFReplacer: public ExprMutator {
	PrimExpr VisitExpr_(const CallNode* op) override {
	  if (this->patterns_map->find(op) != this->patterns_map->end()) {
	    auto pattern = this->patterns_map->find(op)->second;
	    Array<PrimExpr> args;
	    // Skip the last dimension as that's the variant dimension
	    // we handle after the loop
	    for (auto dim: orig_idx_dims) {
	      std::cout << "[D] OrigDim " << dim->name << " " << dim.get() << std::endl;
	    }
	    for (size_t i = 0; i < cache_idx_dims.size() - 1; ++i) {
	      auto dim = cache_idx_dims[i];
	      std::cout << "[D] Dim " << i << " " << dim->name << " " << dim.get() << std::endl;
	      if (!orig_idx_dims.Contains(dim)) {
		// This is newly added dimensions, corresponding to an
		// index of the original tensor. For this, we need to
		// index by the IV corresponding to this dimension.

		if (orig->dimensions.Contains(dim)) {
		  args.push_back(orig->parameters[orig->dimensions.GetIdx(dim)]);
		  std::cout << "[D]   Arg1 " << orig->parameters[orig->dimensions.GetIdx(dim)] << std::endl;
		}
		else {
		  new_param_dims.push_back(dim);
		  Var new_param = Var("p" + std::to_string(i), op->args[i]->dtype);
		  new_params.push_back(new_param);
		  args.push_back(new_param);
		  std::cout << "[D]   Arg2 " << new_param << std::endl;
		}
	      }
	      else {
		// Here we leave the argument intact, for the case
		// where the dimension is left unmodified by the
		// transform.
		args.push_back(op->args[i]);
		std::cout << "[D]   Arg3 " << op->args[i] << std::endl;
	      }
	    }
	    args.push_back(pattern->idx);
	    PrimExpr new_call = CallNode::make(op->dtype, this->cache->op->name, args, op->call_type,
					       this->cache->op, this->cache->value_index);
	    std::cout << "[CRO]  Replacing " << GetRef<PrimExpr>(op) << " " << new_call << std::endl;
	    return new_call;
	  }
	  else return ExprMutator::VisitExpr_(op);
	}

      public:
	UFReplacer(const ComputeOpNode* compute_op_, const AccessToPatternMap* patterns_map_,
		   Tensor cache_, Array<Dimension> cache_idx_dims_, Array<Dimension> orig_idx_dims_) :
	  compute_op(compute_op_), patterns_map(patterns_map_), cache(cache_),
	  cache_idx_dims(cache_idx_dims_), orig_idx_dims(orig_idx_dims_) {}

	UninterpFun replace(UninterpFun orig_) {
	  this->orig = orig_;
	  this->new_param_dims.resize(0);
	  this->new_params.resize(0);

	  PrimExpr body = this->VisitExpr(orig->body);
	  Array<Var> parameters = Array<Var>(orig->parameters);
	  Array<Dimension> dimensions = Array<Dimension>(orig->dimensions);
	  for (size_t i = 0; i < new_params.size(); ++i) {
	    parameters.push_back(new_params[i]);
	    dimensions.push_back(new_param_dims[i]);
	  }
	  return UninterpFunNode::make(orig->fname, orig->range, dimensions, parameters, body);
	}

	const ComputeOpNode* compute_op;
	const AccessToPatternMap* patterns_map;
	Tensor cache;
	Array<Dimension> cache_idx_dims;
	Array<Dimension> orig_idx_dims;
	UninterpFun orig;
	Array<Dimension> new_param_dims;
	Array<Var> new_params;
      };

      auto compute_op = reader.as<ComputeOpNode>();
      CHECK(compute_op) << "Other reader ops not supported yet";

      ExprReplacer expr_replacer(compute_op, patterns_map, cache, cache_idx_dims, orig_idx_dims);
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
	for (auto e: compute_op->body) {
	  PrimExpr new_expr = expr_replacer(e);
	  arr.push_back(new_expr);
	}
      }

      // TODO(ppf): Create new UFuns here instead of just mutating
      // bodies as UFuns may be shared across multiple ops
      UFReplacer uf_replacer(compute_op, patterns_map, cache, cache_idx_dims, orig_idx_dims);
      Array<UninterpFun> new_index_expressions;
      for (size_t i = 0; i < compute_op->index_expressions.size(); ++i) {
	UninterpFun ufun = compute_op->index_expressions[i];
	UninterpFun new_fun = uf_replacer.replace(ufun);
	new_index_expressions.push_back(new_fun);
      }
      const_cast<ComputeOpNode*>(compute_op)->set_index_expressions(new_index_expressions);

      for (auto iv: compute_op->axis) {
	if (auto call = iv->dom->extent.as<CallNode>()) {
	  if (call->func.as<UninterpFunNode>()) {
	    const_cast<CallNode*>(call)->func = uf_replacer.replace(Downcast<UninterpFun>(call->func));
	  }
	}
      }
      if (!arr.same_as(compute_op->body)) {
	return ComputeOpNode::make(compute_op->name, compute_op->tag, compute_op->attrs, compute_op->axis,
				   compute_op->output_shape_storage, compute_op->index_variables,
				   compute_op->index_expressions, compute_op->loop_dimensions,
				   compute_op->index_dimensions, compute_op->root_index_dimensions, arr);
      } else {
	return reader;
      }
    }

    Tensor Schedule::cache_read_opaque(const Tensor& tensor,
				       const std::string& scope,
				       const Array<Operation>& readers) {
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
      }
      else {
	original_loop_axis = placeholder_op->axis;
	original_loop_dimensions = placeholder_op->loop_dimensions;
	original_index_expressions = placeholder_op->index_expressions;
	original_index_dimensions = placeholder_op->index_dimensions;
	original_root_index_dimensions = placeholder_op->self_index_dimensions;
      }

      std::cout << "[OD] " << original_loop_dimensions.size() << std::endl;

      AccessPatternCollector collector(tensor, original_root_index_dimensions, readers);
      collector.collect();
      PatternsSet patterns = collector.access_patterns;
      AccessToPatternMap access_to_pattern_map = collector.access_to_pattern_map;

      /************* Create the cache stage *************/
      // Create the body of the cache stage
      std::string cache_name = tensor->op->name + "." + scope;
      std::string cache_tag = {};
      Map<std::string, ObjectRef> cache_attrs = {};

      Array<IterVar> cache_axis;
      {
	std::unordered_map<const VarNode*, PrimExpr> replace_map;
	for (size_t i = 0; i < original_loop_axis.size(); ++i) {
	  auto lv = original_loop_axis[i];
	  Var var = Var("lv" + std::to_string(i), DataType::Int(32));
	  VarReplacer replacer(replace_map);
	  cache_axis.push_back(IterVarNode::make(Range::make_by_min_extent(replacer(lv->dom->min), replacer(lv->dom->extent)),
						 var, lv->iter_type, lv->thread_tag));
	  replace_map[lv->var.get()] = var;
	}
	cache_axis.push_back(IterVarNode::make(Range(0, static_cast<int>(patterns.size())), Var("var", DataType::Int(32)),
					       IterVarType::kDataPar, ""));
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
	  cache_index_variables.push_back(IterVarNode::make(uif->range, Var("iv" + std::to_string(i), DataType::Int(32)),
							    IterVarType::kDataPar, ""));
	}

	cache_index_expressions = Array<UninterpFun>(original_index_expressions);
	cache_index_dimensions = Array<Dimension>(original_index_dimensions);
      }

      Array<Dimension> cache_loop_dimensions;
      Array<Dimension> cache_root_index_dimensions;
      {
	cache_loop_dimensions = Array<Dimension>(original_loop_dimensions);
	// cache_root_index_dimensions = Array<Dimension>(original_root_index_dimensions);
	cache_root_index_dimensions = Array<Dimension>(original_loop_dimensions);
	auto variant_dim = DimensionNode::make("variants", DimensionNode::kRangeDim);
	cache_loop_dimensions.push_back(variant_dim);
	cache_root_index_dimensions.push_back(variant_dim);
      }

      PatternsVec patterns_vec;
      for (auto pattern: patterns) {
	pattern->idx = patterns_vec.size();
	patterns_vec.push_back(pattern);
      }

      Array<PrimExpr> cache_body = { CacheBodyBuilder(tensor, original_root_index_dimensions,
						      patterns_vec, cache_index_variables,
						      cache_axis, cache_index_dimensions, cache_loop_dimensions) };

      for (auto dim: cache_root_index_dimensions) {
	std::cout << "[CRO] Cache root dim " << dim->name << std::endl;
      }

      Tensor cache = ComputeOpNode::make(cache_name, cache_tag, cache_attrs, cache_axis, cache_shape,
					 cache_index_variables, cache_index_expressions, cache_loop_dimensions,
					 cache_index_dimensions, cache_root_index_dimensions, cache_body).output(0);

      /************* Replace reader inputs *************/
      std::unordered_map<Tensor, Tensor> vmap;
      std::unordered_map<Tensor, Tensor> rvmap;
      std::cout << "[OD2] " << original_loop_dimensions.size() << std::endl;
      for (Operation op : readers) {
	Stage s = operator[](op);
	Operation repl_op = ReplaceInputs(s->op, &access_to_pattern_map, cache,
					  cache_root_index_dimensions, original_loop_dimensions);
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
      stages->data.insert(stages->data.begin() + pos + 1,
			  cache_stage);
      (*this)->stage_map.Set(cache->op, cache_stage);
      // Update group
      cache_stage->group = op_stage->group;
      if (cache_stage->group.defined()) {
	++cache_stage->group->num_child_stages;
      }
      return cache;

    }
  }
}
