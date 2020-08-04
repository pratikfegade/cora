#include <tvm/te/schedule.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/expr.h>

#include "graph.h"
#include "schedule_utils.h"

namespace tvm {
namespace te {
Array<Array<Operation>, Array<Operation>> LowerDynamicBatching(Array<Operation> outputs,
                                                               Var num_nodes, Var num_batches,
                                                               Var max_batch_len, Var max_child_num,
                                                               Var max_int_idx);
}
}  // namespace tvm
