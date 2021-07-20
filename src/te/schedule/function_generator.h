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

class AFunGenerator {
 public:
  AFunGenerator(const Schedule& sch_) : sch(sch_) {}

  Stmt GenerateAndSetAFuns(Map<Buffer, Buffer>* p_buffer_map);

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

  UninterpFun set_afun(Modes layout, int idx, UninterpFun a_fun_shell,
                       Map<Buffer, Buffer>* p_buffer_map, AllocationAggregator* p_host_agg,
                       AllocationAggregator* p_dev_agg);

  Schedule sch;
  std::unordered_map<FunKey, UninterpFun, FunKeyHasher, FunKeyEquality> dim_afun_map;
  Array<Stmt> stmts;
  int count{0};
};

class RaggedFusionBoundStmtsGenerator : public StmtExprMutator {
 public:
  RaggedFusionBoundStmtsGenerator(const Schedule& sch_,
                                  const std::unordered_map<IterVar, Range>& dom_map_)
      : sch(sch_), dom_map(dom_map_), count(0) {}

  Stmt Generate(Array<ObjectRef>* p_non_negative_objects, Map<Buffer, Buffer>* p_buffer_map);

 private:
  PrimExpr root_ivs_fused(Stage& stage, Array<IterVar> fused_ivs);

  Stmt generate_fusion_statements(Stage& stage, const RaggedFuseNode* rel,
                                  Array<ObjectRef>* p_non_negative_objects,
                                  Map<Buffer, Buffer>* p_buffer_map,
                                  AllocationAggregator* p_host_agg,
                                  AllocationAggregator* p_dev_agg);

  Array<PrimExpr> get_iter_var_values(Array<IterVar> vars, Stage& stage);

  const Schedule& sch;
  const std::unordered_map<IterVar, Range>& dom_map;
  int count;
};

class FunctionGenerator {
 public:
  FunctionGenerator(const Schedule& sch_, const std::unordered_map<IterVar, Range>& dom_map_)
      : sch(sch_), dom_map(dom_map_) {}

  void GenerateAFunctions();

  void GenerateFusionFunctions();

  Stmt CreateBody(Stmt body);

 private:
  const Schedule& sch;
  const std::unordered_map<IterVar, Range>& dom_map;
  Map<Buffer, Buffer> buffer_map;
  Array<ObjectRef> non_negative_objects;
  Stmt afun_stmt;
  Stmt ffun_stmt;
};

}  // namespace te
}  // namespace tvm

#endif  // TVM_TE_FUNCTION_GENERATOR_H_
