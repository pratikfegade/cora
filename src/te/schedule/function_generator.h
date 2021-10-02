#ifndef TVM_TE_FUNCTION_GENERATOR_H_
#define TVM_TE_FUNCTION_GENERATOR_H_

#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <tvm/te/schedule_pass.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/tir/stmt_functor.h>

#include <set>
#include <unordered_map>

namespace tvm {
namespace te {

class AllocationAggregator {
 public:
  AllocationAggregator(std::string aggregate_name_, DataType dtype_)
      : aggregate_name(aggregate_name_),
        dtype(dtype_),
        aggregate_buffer_var(aggregate_name_, DataType::Handle()),
        aggregate_allocated_size(0) {}

  Buffer create_buffer(Array<PrimExpr> extents, DataType buf_dtype, std::string name);

  Buffer aggregate_buffer();

  PrimExpr aggregate_size() const { return aggregate_allocated_size; }

 private:
  std::string aggregate_name;
  DataType dtype;
  Var aggregate_buffer_var;
  PrimExpr aggregate_allocated_size;
};

class AggregatorPair {
 public:
  AggregatorPair(bool distinct_device_)
      : distinct_device(distinct_device_),
        host_agg("host_buf", DataType::Int(32)),
        dev_agg("dev_buf", DataType::Int(32)) {}

  std::pair<Buffer, Buffer> create_buffer_pair(Array<PrimExpr> extents, DataType buf_dtype,
                                               std::string name);

  std::pair<Buffer, Buffer> aggregate_buffers();

  PrimExpr aggregate_size() const { return host_agg.aggregate_size(); }

  PrimExpr current_device_buffer_size() const { return dev_agg.aggregate_size(); }

 private:
  bool distinct_device;
  AllocationAggregator host_agg;
  AllocationAggregator dev_agg;
};

class AFunctionGenerator {
 public:
  AFunctionGenerator(const Schedule& sch_, Map<Buffer, Buffer>* p_buffer_map_,
                     AggregatorPair* p_agg_pair_, bool debug_fill_function_bodies_,
                     Array<Buffer> afuns_needed_for_)
      : sch(sch_),
        buffer_map(*p_buffer_map_),
        agg_pair(*p_agg_pair_),
        debug_fill_function_bodies(debug_fill_function_bodies_),
        afuns_needed_for(afuns_needed_for_) {}

  Stmt Generate();

  struct FunKey {
    Dimension dimension;
    std::multiset<const Object*> dependent_dimensions;
  };

 private:
  class FunKeyHasher {
   public:
    size_t operator()(const FunKey& pattern) const;
  };

  class FunKeyEquality {
   public:
    bool operator()(const FunKey& p1, const FunKey& p2) const;
  };

  UninterpFun set_afun(Modes layout, int idx, UninterpFun a_fun_shell);

  Schedule sch;
  Map<Buffer, Buffer>& buffer_map;
  AggregatorPair& agg_pair;
  bool debug_fill_function_bodies;
  Array<Buffer> afuns_needed_for;
  std::unordered_map<FunKey, UninterpFun, FunKeyHasher, FunKeyEquality> dim_afun_map;
  Array<Stmt> stmts;
  int count{0};
};

class FusionFunctionGenerator : public StmtExprMutator {
 public:
  FusionFunctionGenerator(const Schedule& sch_, const std::unordered_map<IterVar, Range>& dom_map_,
                          Map<Stage, Modes>& root_layout_map_,
                          const std::vector<Stage>& stages_to_generate_for_,
                          Array<ObjectRef>* p_non_negative_objects_,
                          Map<Buffer, Buffer>* p_buffer_map_, AggregatorPair* p_agg_pair_,
                          bool debug_fill_function_bodies_)
      : sch(sch_),
        dom_map(dom_map_),
        root_layout_map(root_layout_map_),
        stages_to_generate_for(stages_to_generate_for_),
        non_negative_objects(*p_non_negative_objects_),
        buffer_map(*p_buffer_map_),
        agg_pair(*p_agg_pair_),
        debug_fill_function_bodies(debug_fill_function_bodies_),
        count(0) {
    for (auto it : root_layout_map_) {
      /* std::cout << "[MAPMAP] " << it.first << " " << it.second << std::endl; */
      root_layout_map.Set(it.first, it.second);
    }
  }

  Stmt Generate();

  Stmt generate_fusion_statements(Stage& stage, const RaggedFuseNode* rel);

  Stmt generate_fusion_statements(Stage& stage, const RaggedDimensionFuseNode* rel);

  const Schedule& sch;
  const std::unordered_map<IterVar, Range>& dom_map;
  Map<Stage, Modes>& root_layout_map;
  const std::vector<Stage> stages_to_generate_for;
  Array<ObjectRef>& non_negative_objects;
  Map<Buffer, Buffer>& buffer_map;
  AggregatorPair& agg_pair;
  bool debug_fill_function_bodies;

 private:
  int count;
};

class FusionFunctionSimplifier : public StmtExprMutator {
 public:
  FusionFunctionSimplifier(const Schedule& sch_, const std::unordered_map<IterVar, Range>& dom_map_)
      : sch(sch_), dom_map(dom_map_) {}

  Stmt Simplify(Stmt body, std::vector<Stage>& stages_to_generate_fusion_funcs_for);

  PrimExpr Simplify(PrimExpr e) { return this->VisitExpr(e); }

  Range Simplify(Range r) {
    return Range::make_by_min_extent(this->VisitExpr(r->min), this->VisitExpr(r->extent));
  }

 private:
  PrimExpr VisitExpr_(const CallNode* op) override;

  PrimExpr VisitExpr_(const FuseSelectNode* op) override;

  const Schedule& sch;
  const std::unordered_map<IterVar, Range>& dom_map;
  std::unordered_map<const Object*, UninterpFun> fsub;
};

class FunctionGenerator {
 public:
  FunctionGenerator(const Schedule& sch_, const std::unordered_map<IterVar, Range>& dom_map_,
                    bool distinct_device_, bool debug_fill_function_bodies_,
                    Array<Buffer> afuns_needed_for_)
      : sch(sch_),
        dom_map(dom_map_),
        agg_pair(distinct_device_),
        debug_fill_function_bodies(debug_fill_function_bodies_),
        afuns_needed_for(afuns_needed_for_) {
    for (auto s : sch->stages) {
      for (auto rel : s->dim_relation_graph->relations) {
        if (rel.as<RaggedDimensionFuseNode>()) {
          /* std::cout << "[GFS] Map " << s << " " << s->op->output_layout(0) << std::endl; */
          root_layout_map.Set(s, s->op->output_layout(0));
        }
      }
    }
  }

  Stmt SimplifyFusionFunctions(Stmt body);

  void GenerateAFunctions();

  void GenerateFusionFunctions();

  Stmt CreateBody(Stmt body);

  PrimExpr GetCurrentAggregateBufferSize() { return agg_pair.current_device_buffer_size(); }

 private:
  const Schedule& sch;
  const std::unordered_map<IterVar, Range>& dom_map;
  AggregatorPair agg_pair;
  bool debug_fill_function_bodies;
  Array<Buffer> afuns_needed_for;
  Map<Buffer, Buffer> buffer_map;
  Array<ObjectRef> non_negative_objects;
  std::vector<Stage> stages_to_generate_fusion_funcs_for;
  Map<Stage, Modes> root_layout_map;
  Stmt afun_stmt;
  Stmt ffun_stmt;
};

}  // namespace te
}  // namespace tvm

#endif  // TVM_TE_FUNCTION_GENERATOR_H_
