#include <tvm/ir/attrs.h>
#include <tvm/te/operation.h>
#include <tvm/te/schedule.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_equality.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/tir/modes.h>
#include <tvm/tir/stmt_functor.h>

#include <unordered_set>

#include "../../tir/ir/var_replacer.h"
#include "message_passing.h"

namespace tvm {
namespace te {

bool verify_itervar_order(const Stage& stage, const Array<IterVar>& order);

bool verify_dimension_order(const Stage& stage, const Array<Dimension>& order);

std::pair<UninterpFun, UninterpFun> GetLFunction(StageNode* self, Dimension dim,
                                                 bool want_loop_l_fun, int value_index = -1);

}  // namespace te
}  // namespace tvm
