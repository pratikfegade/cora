#include "schedule_utils.h"

#include <tvm/te/schedule.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/expr.h>

#include "graph.h"

namespace tvm {
namespace te {
bool isCudaThread(const IterVar& iv) { return isCudaThread(iv->var->name_hint); }

bool isCudaThread(const std::string& name) {
  return name == "blockIdx.x" || name == "blockIdx.y" || name == "blockIdx.z" ||
         name == "threadIdx.x" || name == "threadIdx.y" || name == "threadIdx.z";
}

ReadGraph GetReadGraph(Schedule& sch, bool includeUnemittedInputs, bool print) {
  static Array<Operation> roots;
  roots.resize(0);
  for (Operation op : sch->outputs) {
    roots.push_back(sch->stage_map[op]->op);
  }
  return CreateReadGraph(roots, includeUnemittedInputs, print);
}

FeedGraph GetFeedGraph(Schedule& sch, bool includeUnemittedInputs) {
  return CreateFeedGraph(GetReadGraph(sch, includeUnemittedInputs));
}

Array<Tensor> RemapTensor(ScheduleNode* self, const Array<Tensor>& arr) {
  self->InitCache();
  const auto& op2stage_cache = self->op2stage_cache_;
  Array<Tensor> ret;
  for (Tensor t : arr) {
    if (!op2stage_cache.count(t->op.get())) {
      CHECK(self->stage_map.count(t->op)) << "Given tensor is not in the schedule plan " << t->op;
      t = self->stage_map[t->op]->op.output(t->value_index);
    }
    ret.push_back(t);
  }
  return ret;
}

bool CheckSchedule(Schedule& sch, const std::string& caller, bool print) {
  // std::cout << "[YOYOYOSDFVBOTOTO]" << std::endl;
  // for (const auto& s : sch->stages) {
  //   std::cout << "[SK] " << s << " " << s->op << std::endl;
  // }

  sch->InvalidateCache();
  sch->InitCache();

  auto rg = GetReadGraph(sch, print);
  for (auto it : rg) {
    Operation op = it.first;
    Array<Tensor> reads = it.second;
    CHECK(sch->op2stage_cache_.count(op.get())) << op << " " << caller;
    for (auto t : reads) {
      CHECK(sch->op2stage_cache_.count(t->op.get())) << t->op << " " << op << " " << caller;
    }
  }

  auto fg = CreateFeedGraph(rg);
  for (auto it : fg) {
    CHECK(sch->op2stage_cache_.count(it.first->op.get())) << it.first->op << " " << caller;
    for (auto op : it.second) {
      CHECK(sch->op2stage_cache_.count(op.get())) << op << " " << it.first->op << " " << caller;
    }
  }

  // for (auto s : sch->stages) {
  //   if (s->attach_type == kInlinedAlready) continue;
  //   if (s->is_output || s->op.as<PlaceholderOpNode>()) continue;
  //   for (int i = 0; i < s->op->num_outputs(); ++i) {
  //     Tensor t = s->op.output(i);
  //     if (fg.count(t) == 0) {
  //       std::cout << "[YERROR] " << err << " " << t << " " << s->op << std::endl;
  //     }
  //   }
  // }
  return true;
}

// Replace data flow appears in all stages given the tensor change.
// Also update vmap if subsequent dataflow need to be replaced.
// Need to keep an update to the date transitive closure property on the vmap by a reverse map.
void ReplaceDataFlow(const Array<Stage>& stages, Map<FunctionRef, CacheInfo> cacheMappings,
                     std::unordered_map<Tensor, Tensor>* vmap,
                     std::unordered_map<Tensor, Tensor>* rvmap,
                     std::unordered_set<const OperationNode*> to_skip) {
  for (Stage s : stages) {
    if (to_skip.count(s->op.as<OperationNode>())) continue;
    Operation op = s->op->ReplaceInputs(s->op, *vmap);
    if (!op.same_as(s->op)) {
      // std::cout << "[RDF]   Replacing " << s->op << " with " << op << std::endl;
      for (int i = 0; i < op->num_outputs(); ++i) {
        auto it = rvmap->find(s->op.output(i));
        if (it != rvmap->end()) {
          (*vmap)[it->second] = op.output(i);
        } else {
          (*vmap)[s->op.output(i)] = op.output(i);
          (*rvmap)[op.output(i)] = s->op.output(i);
        }
      }
      if (cacheMappings.count(s->op)) {
        cacheMappings.Set(op, cacheMappings.at(s->op));
      }
      s->op = op;
    }
  }
}

}  // namespace te
}  // namespace tvm
