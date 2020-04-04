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
    Map<Dimension, PrimExpr> args;
    const CallNode* original_access;
    const ComputeOpNode* reader_op;
    int idx;

    class Hasher {
    public:
      size_t operator()(const AccessPattern* pattern) const {
	AttrsHash hasher;
	using std::hash;
	size_t h = 0;
	for (auto it: pattern->args) {
	  Dimension d = it.first;
	  PrimExpr i = it.second;
	  h += hasher(d) + hasher(i);
	}
	return h;
      }
    };
    class Equality {
    public:
      bool operator()(const AccessPattern* p1, const AccessPattern* p2) const {
	AttrsEqual equals;
	for (auto it1: p1->args) {
	  Dimension d1 = it1.first;
	  PrimExpr idx1 = it1.second;
	  if (p2->args.count(d1)) {
	    PrimExpr idx2 = p2->args.at(d1);
	    if (!equals(idx1, idx2)) {
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
	    // std::cout << "[APC] Found access " << GetRef<PrimExpr>(op) << std::endl;
	    AccessPattern* ap = new AccessPattern();
	    for (size_t i = 0; i < op->args.size(); ++i) {
	      auto expanded = this->ExpandToLoopVars(op->args[i], reader_op);
	      ap->args.Set(tensor_index_dims[i], expanded);
	    }
	    ap->original_access = op;
	    ap->reader_op = reader_op;
	    this->access_patterns->insert(ap);
	    (*this->access_to_pattern_map)[op] = ap;
	  }
	}
      }

    public:
      ExprAccessPatternCollector(const Tensor& tensor_, PatternsSet* access_patterns_,
				 AccessToPatternMap* access_to_pattern_map_, const ComputeOpNode* reader_op_) :
	tensor(tensor_), access_patterns(access_patterns_), access_to_pattern_map(access_to_pattern_map_), reader_op(reader_op_) {
	if (auto op = tensor->op.as<ComputeOpNode>()) {
	  this->tensor_index_dims = op->self_index_dimensions;
	}
	else if (auto op = tensor->op.as<PlaceholderOpNode>()) {
	  this->tensor_index_dims = op->index_dimensions;
	}
	else {
	  CHECK(false) << "Cannot only cache Compute and Plcceholder operations";
	}
      }

      const Tensor& tensor;
      PatternsSet* access_patterns;
      AccessToPatternMap* access_to_pattern_map;
      const ComputeOpNode* reader_op;
      Array<Dimension> tensor_index_dims;
   };

    void collectAccesPatterns() {
      for (auto reader: this->readers) {
	if (auto reader_op = reader.as<ComputeOpNode>()) {
	  for (auto body_expr: reader_op->body) {
	    ExprAccessPatternCollector exprCollector(this->tensor, &(this->access_patterns),
						     &(this->access_to_pattern_map), reader_op);
	    exprCollector(body_expr);
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

    AccessPatternCollector(const Tensor& tensor_, const Array<Operation>& readers_) :
      tensor(tensor_), readers(readers_) {}

    PatternsSet access_patterns;
    AccessToPatternMap access_to_pattern_map;
    const Tensor& tensor;
    const Array<Operation>& readers;
  };

  class TranslateVarsCrossStages: public ExprMutator {
    PrimExpr VisitExpr_(const VarNode* op) override {

      Dimension var_dim;
      for (size_t i = 0; i < reader_op->index_variables.size(); ++i) {
	if (op == reader_op->index_variables[i]->var.get()) {
	  var_dim = reader_op->index_dimensions[i];
	}
      }

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

  PrimExpr CacheBodyBuilder(const PatternsVec& patterns_vec, Array<IterVar> index_variables,
			    Array<IterVar> loop_variables, Array<Dimension> index_dimensions,
			    Array<Dimension> loop_dimensions) {
    const Var variant_loop_var = loop_variables[loop_variables.size() - 1]->var;

    PrimExpr body = PrimExpr(0);
    for (size_t i = 0; i < patterns_vec.size(); ++i) {
      AccessPattern* pattern = patterns_vec[i];
      body = if_then_else(variant_loop_var == static_cast<int>(i),
			  TranslateVarsCrossStages(pattern->original_access, pattern->reader_op, index_variables,
						   loop_variables, index_dimensions, loop_dimensions)
			  (GetRef<PrimExpr>(pattern->original_access)),
			  body);
    }

    return body;
  }

  Operation ReplaceInputs(Operation reader, const AccessToPatternMap* patterns_map, Tensor cache) {
    class Replacer: public ExprMutator {
      PrimExpr VisitExpr_(const CallNode* op) override {
	if (this->patterns_map->find(op) != this->patterns_map->end()) {
	  auto pattern = this->patterns_map->find(op)->second;
	  Array<PrimExpr> args;
	  for (auto iv: this->compute_op->axis) {
	    args.push_back(iv->var);
	  }
	  args.push_back(pattern->idx);
	  return CallNode::make(op->dtype, this->cache->op->name, args, op->call_type,
				this->cache->op, this->cache->value_index);
	}
	else return ExprMutator::VisitExpr_(op);
      }

    public:
      Replacer(const ComputeOpNode* compute_op_, const AccessToPatternMap* patterns_map_, Tensor cache_) :
	compute_op(compute_op_), patterns_map(patterns_map_), cache(cache_) {}

      const ComputeOpNode* compute_op;
      const AccessToPatternMap* patterns_map;
      Tensor cache;
    };

    auto compute_op = reader.as<ComputeOpNode>();
    CHECK(compute_op) << "Other reader ops not supported yet";

    Replacer replacer(compute_op, patterns_map, cache);
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
      for (auto e: compute_op->body) {
	PrimExpr new_expr = replacer(e);
	std::cout << "New body " << new_expr << std::endl;
	arr.push_back(new_expr);
      }
    }
    if (!arr.same_as(compute_op->body)) {
      return ComputeOpNode::make(compute_op->name, compute_op->tag, compute_op->attrs, compute_op->axis,
				 compute_op->output_shape_storage, compute_op->index_variables,
				 compute_op->index_expressions, compute_op->loop_dimensions,
				 compute_op->index_dimensions, compute_op->self_index_dimensions, arr);
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
    AccessPatternCollector collector(tensor, readers);
    collector.collect();
    PatternsSet patterns = collector.access_patterns;
    AccessToPatternMap access_to_pattern_map = collector.access_to_pattern_map;
    std::cout << "Patterns: " << patterns.size() << std::endl;

    /************* Create the cache stage *************/
    Array<IterVar> original_loop_axis;
    Array<Dimension> original_loop_dimensions;
    if (compute_op) {
      original_loop_axis = compute_op->axis;
      original_loop_dimensions = compute_op->loop_dimensions;
    }
    else {
      original_loop_axis = placeholder_op->axis;
      original_loop_dimensions = placeholder_op->loop_dimensions;
    }

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
					       var, lv->iter_type, lv->loop_axis, lv->thread_tag));
	std::cout << "Original axis: " << lv << std::endl;
	std::cout << "Cache extent: " << lv->dom << std::endl;
	replace_map[lv->var.get()] = var;
      }
      cache_axis.push_back(IterVarNode::make(Range(0, static_cast<int>(patterns.size())), Var("var", DataType::Int(32)),
					     IterVarType::kDataPar, ""));
    }

    Array<PrimExpr> cache_shape;
    {
      for (size_t i = 0; i < original_loop_axis.size(); ++i) {
	// TODO(ppf):: Take actual loop dim extents
	cache_shape.push_back(1000);
      }
      // Pattern dimension
      cache_shape.push_back(static_cast<int>(patterns.size()));
    }

    Array<IterVar> cache_index_variables;
    Array<UninterpFun> cache_index_expressions;
    Array<Dimension> cache_index_dimensions;
    {
      if (compute_op) {
	for (size_t i = 0; i < compute_op->index_variables.size(); ++i) {
	  auto iv = compute_op->index_variables[i];
	  cache_index_variables.push_back(IterVarNode::make(iv->dom, Var("iv" + std::to_string(i), DataType::Int(32)),
							    iv->iter_type, iv->thread_tag));
	}

	cache_index_expressions = Array<UninterpFun>(compute_op->index_expressions);
	cache_index_dimensions = Array<Dimension>(compute_op->index_dimensions);
      }
      else {
	for (size_t i = 0; i < placeholder_op->index_expressions.size(); ++i) {
	  auto uif = placeholder_op->index_expressions[i];
	  cache_index_variables.push_back(IterVarNode::make(uif->range, Var("iv" + std::to_string(i), DataType::Int(32)),
							    IterVarType::kDataPar, ""));
	}

	cache_index_expressions = Array<UninterpFun>(placeholder_op->index_expressions);
	cache_index_dimensions = Array<Dimension>(placeholder_op->index_dimensions);
      }
    }

    Array<Dimension> cache_loop_dimensions;
    Array<Dimension> cache_self_index_dimensions;
    {
      cache_loop_dimensions = Array<Dimension>(original_loop_dimensions);
      cache_self_index_dimensions = Array<Dimension>(original_loop_dimensions);
      auto variant_dim = DimensionNode::make("variants", DimensionNode::DimensionType::kRangeDim);
      cache_loop_dimensions.push_back(variant_dim);
      cache_self_index_dimensions.push_back(variant_dim);
    }

    PatternsVec patterns_vec;
    for (auto pattern: patterns) {
      pattern->idx = patterns_vec.size();
      patterns_vec.push_back(pattern);
    }

    Array<PrimExpr> cache_body = { CacheBodyBuilder(patterns_vec, cache_index_variables, cache_axis,
						    cache_index_dimensions, cache_loop_dimensions) };

    Tensor cache = ComputeOpNode::make(cache_name, cache_tag, cache_attrs, cache_axis, cache_shape,
				       cache_index_variables, cache_index_expressions, cache_loop_dimensions,
				       cache_index_dimensions, cache_self_index_dimensions, cache_body).output(0);

    /************* Replace reader inputs *************/
    // for (auto reader: readers) {
    //   ReplaceInputs(reader, &access_to_pattern_map, cache);
    // }

    // return cache;







  std::unordered_map<Tensor, Tensor> vmap;
  std::unordered_map<Tensor, Tensor> rvmap;
  for (Operation op : readers) {
    Stage s = operator[](op);
    Operation repl_op = ReplaceInputs(op, &access_to_pattern_map, cache);
    CHECK(!repl_op.same_as(s->op))
        << "Cannot find " << tensor
        << " in the inputs of " << s->op;
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
