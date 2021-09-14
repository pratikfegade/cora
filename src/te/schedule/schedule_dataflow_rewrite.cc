/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file schedule_dataflow_rewrite.cc
 */
#include <tvm/ir/attrs.h>
#include <tvm/te/operation.h>
#include <tvm/te/schedule.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/tir/stmt_functor.h>

#include <unordered_set>

#include "../../arith/compute_expr.h"
#include "../../tir/ir/var_replacer.h"
#include "../../tir/pass/ir_util.h"
#include "message_passing.h"
#include "schedule_utils.h"

namespace tvm {
namespace te {
PrimExpr InjectPredicate(const Array<PrimExpr>& predicates, PrimExpr body) {
  using tir::ReduceNode;
  using tir::SelectNode;
  if (predicates.size() == 0) return body;
  const ReduceNode* reduce = body.as<ReduceNode>();
  if (reduce) {
    auto n = make_object<ReduceNode>(*reduce);
    n->condition = n->condition && arith::ComputeReduce<tir::AndNode>(predicates, PrimExpr());
    return PrimExpr(n);
  }
  return SelectNode::make(arith::ComputeReduce<tir::AndNode>(predicates, PrimExpr()), body,
                          make_zero(body.dtype()));
}

inline bool ReduceEqual(const tir::ReduceNode* a, const tir::ReduceNode* b) {
  return (a->combiner.same_as(b->combiner)) && (a->source.same_as(b->source)) &&
         (a->axis.same_as(b->axis)) && (a->condition.same_as(b->condition));
}

std::pair<Array<UninterpFun>, Array<UninterpFun>> ExtractUFsFromAxis(Array<IterVar> axis) {
  auto get_or_create_uf = [&](const PrimExpr& expr) {
    if (auto call = expr.as<CallNode>()) {
      // If not, we need to create a dummy uninterp fun
      CHECK(call->func.as<UninterpFunNode>());
      return Downcast<UninterpFun, FunctionRef>(call->func);
    } else if (auto var = expr.as<VarNode>()) {
      return UninterpFunNode::make(var->name_hint, Range(expr, expr), Array<Dimension>(),
                                   Array<Var>(), expr, UninterpFunNode::kLFun);
    } else if (expr.as<IntImmNode>()) {
      return UninterpFunNode::make("imm_uf", Range(expr, expr), Array<Dimension>(), Array<Var>(),
                                   expr, UninterpFunNode::kLFun);
    } else {
      CHECK(false) << expr;
      return NullValue<UninterpFun>();
    }
  };

  Array<UninterpFun> min_ufs;
  Array<UninterpFun> extent_ufs;
  for (auto iv : axis) {
    min_ufs.push_back(get_or_create_uf(iv->dom->min));
    extent_ufs.push_back(get_or_create_uf(iv->dom->extent));
  }

  return std::make_pair(min_ufs, extent_ufs);
}

Tensor Schedule::cache_read(const Tensor& tensor, const std::string& scope,
                            const Array<Operation>& readers, std::string suffix, bool vanilla,
                            Array<Modes> cache_storage_layout, Modes cache_loop_layout,
                            bool axis_mirror_loop_layout) {
  // std::cout << "[CR] Caching " << tensor << " " << cache_loop_layout << std::endl;
  (*this)->InvalidateCache();
  // create identity mapping.
  std::ostringstream os;
  os << tensor->op->name;
  if (tensor->op->num_outputs() != 1) {
    os << ".v" << tensor->value_index;
  }
  os << "." << scope << suffix;

  std::unordered_map<Tensor, Tensor> vsub;
  Stage s = operator[](tensor->op);
  Tensor sugar_tensor = s->op.output(tensor->value_index);
  Tensor cache;
  const ComputeOpNode* compute_op = tensor->op.as<ComputeOpNode>();
  const PlaceholderOpNode* placeholder_op = tensor->op.as<PlaceholderOpNode>();
  if ((compute_op || placeholder_op) && !vanilla) {
    Array<IterVar> axis;
    Array<DimInfo> dim_infos;
    Array<Dimension> self_index_dimensions;
    Array<Modes> storage_layouts = cache_storage_layout;
    Modes loop_layout = cache_loop_layout;
    if (compute_op) {
      axis = compute_op->axis;
      dim_infos = compute_op->all_dimensions;
      self_index_dimensions = compute_op->root_index_dimensions;
      if (storage_layouts.size() == 0) {
        storage_layouts = compute_op->storage_layouts;
      }
      if (!loop_layout.defined()) {
        loop_layout = compute_op->loop_layout_object;
      }
    } else {
      for (const auto& di : placeholder_op->all_dimensions) {
        CHECK(di->dim->isLoopDim());
        axis.push_back(di->iv);
      }
      dim_infos = placeholder_op->all_dimensions;
      self_index_dimensions = placeholder_op->self_index_dimensions;
      if (storage_layouts.size() == 0) {
        storage_layouts = {placeholder_op->layout};
      }

      if (!loop_layout.defined()) {
        CHECK(placeholder_op->layout.defined());
        Array<UninterpFun> l_fun_mins;
        for (auto it : placeholder_op->layout->l_funs) {
          l_fun_mins.push_back(UninterpFunNode::from_constant("z", 0));
        }
        loop_layout = ModesNode::make(
            placeholder_op->layout->dimensions, placeholder_op->layout->l_maxes, l_fun_mins,
            placeholder_op->layout->l_funs, placeholder_op->layout->a_funs, true);
      }
    }
    CHECK(loop_layout.defined());
    // std::cout << "[CR] Caching " << tensor->op << std::endl;
    // for (auto lf: storage_layouts[0]->l_funs) {
    //   std::cout << "[CR]  LF " << lf << std::endl;
    // }

    auto axis_ufs = ExtractUFsFromAxis(axis);

    Array<IterVar> new_axis;
    {
      Array<PrimExpr> args;
      Array<Dimension> arg_dims;
      int i = 0;
      for (const auto& di : dim_infos) {
        IterVar new_iv;

        UninterpFun min_uf = axis_ufs.first[i];
        UninterpFun extent_uf = axis_ufs.second[i];
        if (axis_mirror_loop_layout) {
          min_uf = loop_layout->l_fun_mins[i];
          extent_uf = loop_layout->l_funs[i];
          // std::cout << "[CR]   UFs " << min_uf << " " << extent_uf << std::endl;
        }
        PrimExpr min = min_uf->is_constant()
                           ? min_uf->body
                           : min_uf.MakeCallTo(Array<PrimExpr>(args), Array<Dimension>(arg_dims));
        PrimExpr extent =
            extent_uf->is_constant()
                ? extent_uf->body
                : extent_uf.MakeCallTo(Array<PrimExpr>(args), Array<Dimension>(arg_dims));
        new_iv = IterVarNode::make(Range::make_by_min_extent(min, extent),
                                   Var(di->iv->var->name_hint, DataType::Int(32)), kDataPar);
        new_axis.push_back(new_iv);
        args.push_back(new_iv->var);
        arg_dims.push_back(di->dim);
        ++i;
      }
    }

    {
      PrimExpr pred = IntImm(DataType::Bool(), 1);
      Array<PrimExpr> args;
      for (auto iv : new_axis) {
        args.push_back(iv->var);
      }
      PrimExpr body = sugar_tensor(args);
      Operation cache_op =
          ComputeOpNode::make(os.str(), "", {}, new_axis, self_index_dimensions,
                              sugar_tensor->shape, storage_layouts, loop_layout, {body}, {pred});
      cache = cache_op.output(0);
    }
  } else {
    cache = compute(
        sugar_tensor->shape,
        [&sugar_tensor](const Array<Var>& i) {
          return sugar_tensor(Array<PrimExpr>(i.begin(), i.end()));
        },
        os.str());
  }
  vsub[sugar_tensor] = cache;

  std::unordered_map<Tensor, Tensor> vmap;
  std::unordered_map<Tensor, Tensor> rvmap;
  for (Operation op : readers) {
    Stage s = operator[](op);
    Operation repl_op = s->op->ReplaceInputs(s->op, vsub);

    if (repl_op.same_as(s->op)) {
      if (auto c_op = s->op.as<ComputeOpNode>()) {
        CHECK(!repl_op.same_as(s->op))
            << "Cannot find " << tensor << " in the inputs of " << s->op << " " << c_op->body[0];
      }
    }

    CHECK(!repl_op.same_as(s->op)) << "Cannot find " << tensor << " in the inputs of " << s->op;
    vmap[s->op.output(0)] = repl_op.output(0);
    rvmap[repl_op.output(0)] = s->op.output(0);
    s->op = repl_op;
  }
  ReplaceDataFlow((*this)->stages, (*this)->cacheTensorInfos, &vmap, &rvmap);
  ArrayNode* stages = (*this)->stages.CopyOnWrite();
  Stage op_stage = operator[](tensor->op);
  size_t pos = FindNodeRef(stages, op_stage);
  Stage cache_stage = Stage(cache->op);
  cache_stage.set_scope(scope);
  CHECK_LT(pos, stages->data.size());
  stages->data.insert(stages->data.begin() + pos + 1, cache_stage);
  (*this)->stage_map.Set(cache->op, cache_stage);
  // Update group
  cache_stage->group = op_stage->group;
  if (cache_stage->group.defined()) {
    ++cache_stage->group->num_child_stages;
  }
  return cache;
}

Array<Tensor> ReplaceOriginalOp(Schedule sch, Stage orig_stage, const std::string& scope,
                                Operation cache_op, Operation orig_new_op, size_t tensor_size) {
  Array<Tensor> cache_tensor_list;
  for (size_t i = 0; i < tensor_size; i++) {
    Tensor cache_tensor = cache_op.output(i);
    cache_tensor_list.push_back(cache_tensor);
  }
  // The replace of the dataflow
  std::unordered_map<Tensor, Tensor> vmap;
  std::unordered_map<Tensor, Tensor> rvmap;
  vmap[orig_stage->op.output(0)] = orig_new_op.output(0);
  rvmap[orig_new_op.output(0)] = orig_stage->op.output(0);
  for (size_t i = 0; i < tensor_size; i++) {
    vmap[orig_stage->op.output(0)] = orig_new_op.output(0);
    rvmap[orig_new_op.output(0)] = orig_stage->op.output(0);
  }
  ReplaceDataFlow(sch->stages, sch->cacheTensorInfos, &vmap, &rvmap);
  // mutate orig stage
  orig_stage->op = orig_new_op;
  orig_stage->all_iter_vars = orig_stage->op->root_iter_vars();
  orig_stage->leaf_iter_vars = orig_stage->all_iter_vars;
  orig_stage->relations = Array<IterVarRelation>();
  // create schedule for new cached stage.
  ArrayNode* stages = sch->stages.CopyOnWrite();
  size_t pos = FindNodeRef(stages, orig_stage);
  Stage cache_stage = Stage(cache_op);
  cache_stage.set_scope(scope);
  CHECK_LT(pos, stages->data.size());
  stages->data.insert(stages->data.begin() + pos, cache_stage);
  sch->stage_map.Set(cache_op, cache_stage);
  // Update group
  cache_stage->group = orig_stage->group;
  if (cache_stage->group.defined()) {
    ++cache_stage->group->num_child_stages;
  }
  return cache_tensor_list;
}

template <typename OpType>
void PrepareAxisMapping(Stage orig_stage, OpType* op, std::unordered_set<IterVar>* p_red_axis,
                        Array<IterVar>* p_new_axis, std::unordered_map<IterVar, Range>* p_dom_map,
                        std::unordered_map<const VarNode*, PrimExpr>* p_vsub,
                        std::unordered_map<const VarNode*, PrimExpr>* p_vsub2newvar,
                        std::vector<PrimExpr>* p_predicates) {
  // std::cout << "[PAM] Stage " << orig_stage << " " << op->all_dimensions.size() << std::endl;
  auto& red_axis = *p_red_axis;
  auto& new_axis = *p_new_axis;
  auto& dom_map = *p_dom_map;
  auto& vsub = *p_vsub;
  auto& vsub2newvar = *p_vsub2newvar;
  auto& predicates = *p_predicates;
  arith::Analyzer analyzer;

  for (IterVar iv : op->reduce_axis) {
    red_axis.insert(iv);
  }
  for (IterVar iv : op->axis) {
    dom_map[iv] = iv->dom;
    analyzer.Bind(iv->var, iv->dom);
  }
  // te::PassDownDomain(orig_stage, &dom_map, &analyzer, true);
  {
    // The source->cache
    std::unordered_map<IterVar, PrimExpr> value_map;
    for (auto di : op->all_dimensions) {
      CHECK(di->dim->isLoopDim());
      IterVar iv = di->iv;
      if (red_axis.count(iv)) continue;
      CHECK_EQ(iv->iter_type, kDataPar) << "Can only relayout with in data parallel dimensions";
      VarReplacer replacer(vsub2newvar);
      Range dom = Range::make_by_min_extent(replacer(iv->dom->min), replacer(iv->dom->extent));
      IterVar new_iv = IterVarNode::make(dom, iv->var.copy_with_suffix(".c"), iv->iter_type);
      new_axis.push_back(new_iv);
      if (is_one(dom->min)) {
        value_map[iv] = dom->min;
      } else {
        value_map[iv] = iv->var;
      }
      vsub2newvar[iv->var.get()] = new_iv->var;
    }
    // skip reduction iteration.
    std::unordered_set<IterVar> skip_bound_check;
    for (IterVar iv : op->reduce_axis) {
      skip_bound_check.insert(iv);
    }
    // PassUpIndex(orig_stage, dom_map, &value_map, true);

    predicates =
        MakeBoundCheck(orig_stage, dom_map, {}, {}, {}, value_map, true, skip_bound_check, {}, {});
    // The root axis
    for (IterVar iv : op->axis) {
      if (value_map.count(iv)) {
        vsub[iv->var.get()] = value_map.at(iv);
      }  // to handle tensor axis
    }
  }
}

// Cache write and relayout the data according to loop pattern
Array<Tensor> CacheWriteWithReLayout(Schedule sch, const Array<Tensor>& tensor_array,
                                     const std::string& scope, std::string storage_layout_mode) {
  size_t tensor_size = tensor_array.size();
  sch->InvalidateCache();
  Tensor tensor = tensor_array[0];
  Stage orig_stage = sch[tensor->op];
  const ComputeOpNode* compute = orig_stage->op.as<ComputeOpNode>();

  std::unordered_set<IterVar> red_axis;
  Array<IterVar> new_axis;
  std::unordered_map<IterVar, Range> dom_map;

  std::unordered_map<const VarNode*, PrimExpr> vsub;
  std::unordered_map<const VarNode*, PrimExpr> vsub2newvar;
  std::vector<PrimExpr> predicates;

  PrepareAxisMapping(orig_stage, compute, &red_axis, &new_axis, &dom_map, &vsub, &vsub2newvar,
                     &predicates);

  PrimExpr body;
  Array<PrimExpr> body_list;
  const tir::ReduceNode* first_reduce = nullptr;
  for (auto cbody : compute->body) {
    body = VarReplacer(vsub)(cbody);
    body = InjectPredicate(predicates, body);
    body = VarReplacer(vsub2newvar)(body);
    // Reduce nodes in ONE computeOp must be the same except value_index
    // This is right only if the original body ensures Reduce nodes are the same
    if (body->IsInstance<tir::ReduceNode>()) {
      const tir::ReduceNode* reduce_body = body.as<tir::ReduceNode>();
      if (first_reduce != nullptr) {
        CHECK(ReduceEqual(reduce_body, first_reduce));
        body = tir::ReduceNode::make(first_reduce->combiner, first_reduce->source,
                                     first_reduce->axis, first_reduce->condition,
                                     reduce_body->value_index, first_reduce->dimensions);
      } else {
        first_reduce = reduce_body;
      }
    } else {
      CHECK(first_reduce == nullptr) << "cannot mix reduce and other node in ONE compute bodys";
    }
    body_list.push_back(body);
  }

  PrimExpr pred;
  Array<PrimExpr> pred_list;
  for (auto cpred : compute->pred) {
    pred = VarReplacer(vsub)(cpred);
    pred = InjectPredicate(predicates, pred);
    pred = VarReplacer(vsub2newvar)(pred);
    pred_list.push_back(pred);
  }

  // The reader args
  Array<PrimExpr> args;
  {
    for (auto iv : compute->axis) {
      args.push_back(iv->var);
    }
  }

  Array<PrimExpr> new_shape;
  {
    for (auto iv : compute->axis) {
      new_shape.push_back(VarReplacer(vsub2newvar)(iv->dom->extent));
    }
  }

  Array<Dimension> root_dimensions;
  for (const auto dim : compute->root_index_dimensions) {
    CHECK(!dim->isFunDim());
    root_dimensions.push_back(dim);
  }

  // std::cout << "[CW] Name: " << compute->name << " " << storage_layout_mode << std::endl;
  Array<Modes> new_storage_layouts;
  if (storage_layout_mode == "original") {
    new_storage_layouts = compute->storage_layouts;
  } else if (storage_layout_mode == "loop_layout") {
    for (size_t i = 0; i < compute->num_outputs(); ++i) {
      auto layout = compute->loop_layout_object;
      new_storage_layouts.push_back(ModesNode::make(layout->dimensions, layout->l_maxes,
                                                    layout->l_fun_mins, layout->l_funs,
                                                    layout->a_funs, false));
    }
  } else if (storage_layout_mode == "dense") {
    for (size_t i = 0; i < compute->num_outputs(); ++i) {
      new_storage_layouts.push_back(ModesNode::make(root_dimensions, new_shape, {}, {},
                                                    Map<Dimension, UninterpFun>(), false));
    }
  }

  Operation cache_op = ComputeOpNode::make(
      compute->name + "." + scope, compute->tag, compute->attrs, new_axis, root_dimensions,
      new_shape, new_storage_layouts, compute->loop_layout_object, body_list, pred_list);

  Array<PrimExpr> cache_expr_list;
  for (size_t i = 0; i < tensor_size; i++) {
    Tensor cache_tensor = cache_op.output(i);
    cache_expr_list.push_back(cache_tensor(args));
  }
  Operation orig_new_op = ComputeOpNode::make(
      compute->name, compute->tag, compute->attrs, compute->axis, compute->root_index_dimensions,
      compute->output_shape_storage, compute->storage_layouts, compute->loop_layout_object,
      cache_expr_list, compute->pred);
  auto ret = ReplaceOriginalOp(sch, orig_stage, scope, cache_op, orig_new_op, tensor_size);
  CheckSchedule(sch, "schedule_dataflow_rewrite.cc:377_" + tensor->op->name);
  return ret;
}

// for tensor compute op
Array<Tensor> CacheWriteWithReLayoutTensor(Schedule sch, const Array<Tensor>& tensor_array,
                                           const std::string& scope) {
  size_t tensor_size = tensor_array.size();
  sch->InvalidateCache();
  Tensor tensor = tensor_array[0];
  Stage orig_stage = sch[tensor->op];
  const TensorComputeOpNode* tensor_op = orig_stage->op.as<TensorComputeOpNode>();
  CHECK_EQ(tensor_op->num_outputs(), 1)
      << "cache write only support single output tensor_compute_op";

  std::unordered_set<IterVar> red_axis;
  Array<IterVar> new_axis;
  std::unordered_map<IterVar, Range> dom_map;

  std::unordered_map<const VarNode*, PrimExpr> vsub;
  std::unordered_map<const VarNode*, PrimExpr> vsub2newvar;
  std::vector<PrimExpr> predicates;

  PrepareAxisMapping(orig_stage, tensor_op, &red_axis, &new_axis, &dom_map, &vsub, &vsub2newvar,
                     &predicates);

  for (int i = tensor_op->schedulable_ndim; i < static_cast<int>(tensor_op->axis.size()); ++i) {
    IterVar iv = tensor_op->axis[i];
    IterVar new_iv = IterVarNode::make(iv->dom, iv->var.copy_with_suffix(".c"), iv->iter_type);
    new_axis.push_back(new_iv);
  }
  Array<Region> new_regions;
  for (Region old_region : tensor_op->input_regions) {
    Region region;
    for (Range r : old_region) {
      PrimExpr min = VarReplacer(vsub2newvar)(r->min);
      PrimExpr extent = VarReplacer(vsub2newvar)(r->extent);
      region.push_back(Range::make_by_min_extent(min, extent));
    }
    new_regions.push_back(region);
  }

  Array<PrimExpr> new_scalar_inputs;
  for (PrimExpr old_input : tensor_op->scalar_inputs) {
    new_scalar_inputs.push_back(VarReplacer(vsub2newvar)(old_input));
  }

  Operation cache_op = TensorComputeOpNode::make(tensor_op->name + "." + scope, tensor_op->tag,
                                                 new_axis, tensor_op->reduce_axis,
                                                 tensor_op->schedulable_ndim, tensor_op->intrin,
                                                 tensor_op->inputs, new_regions, new_scalar_inputs);

  // axis will be used in generating compute op
  Array<IterVar> compute_axis = tensor_op->axis;
  for (size_t i = tensor_op->schedulable_ndim; i < tensor_op->axis.size(); ++i) {
    IterVar iv = tensor_op->axis[i];
    IterVar aiv = IterVarNode::make(iv->dom, iv->var, kDataPar);
    compute_axis.Set(i, aiv);
  }

  // The reader args
  Array<PrimExpr> args;
  {
    // cache->compute
    std::unordered_map<IterVar, PrimExpr> value_map;
    for (IterVar iv : compute_axis) {
      value_map[iv] = iv->var;
    }
    PassDownIndex(orig_stage, dom_map, &value_map, true);
    for (IterVar iv : orig_stage->leaf_iter_vars) {
      if (red_axis.count(iv)) continue;
      args.push_back(value_map.at(iv));
    }
    // tensorized region axis
    for (size_t i = tensor_op->schedulable_ndim; i < tensor_op->axis.size(); ++i) {
      IterVar iv = compute_axis[i];
      args.push_back(value_map.at(iv));
    }
  }

  Array<PrimExpr> cache_expr_list;
  for (size_t i = 0; i < tensor_size; i++) {
    Tensor cache_tensor = cache_op.output(i);
    cache_expr_list.push_back(cache_tensor(args));
  }
  Operation orig_new_op =
      ComputeOpNode::make(tensor_op->name, tensor_op->tag, {}, compute_axis, cache_expr_list);
  return ReplaceOriginalOp(sch, orig_stage, scope, cache_op, orig_new_op, tensor_size);
}

Array<Tensor> Schedule::cache_write(const Array<Tensor>& tensor_array, const std::string& scope,
                                    std::string storage_layout_mode) {
  (*this)->InvalidateCache();
  CHECK(tensor_array.size() > 0) << "size of tensor_array must be greater than 0";
  Tensor tensor = tensor_array[0];
  Stage orig_stage = operator[](tensor->op);
  const ComputeOpNode* compute = tensor->op.as<ComputeOpNode>();
  CHECK(static_cast<size_t>(compute->num_outputs()) == tensor_array.size())
      << "size of input tensor list must be same as number of stage outputs";
  for (size_t i = 1; i < tensor_array.size(); i++) {
    Stage tmp_stage = operator[](tensor_array[i]->op);
    CHECK(orig_stage.same_as(tmp_stage)) << "Input tensor list must be generated by ONE computeOp";
  }
  return CacheWriteWithReLayout(*this, tensor_array, scope, storage_layout_mode);
}

Tensor Schedule::cache_write(const Tensor& tensor, const std::string& scope,
                             std::string storage_layout_mode) {
  // support original compute and tensor compute both
  (*this)->InvalidateCache();
  if (tensor->op.as<ComputeOpNode>()) {
    return (CacheWriteWithReLayout(*this, {tensor}, scope, storage_layout_mode))[0];
  } else if (tensor->op.as<TensorComputeOpNode>()) {
    return (CacheWriteWithReLayoutTensor(*this, {tensor}, scope))[0];
  } else {
    LOG(FATAL) << "cache write only take ComputeOp or TensorComputeOp as writers";
    return Tensor();
  }
}

void RebaseNonZeroMinLoop(const Schedule& sch) {
  std::unordered_map<IterVar, IterVar> rebase_map;
  for (Stage s : sch->stages) {
    if (s->attach_type == kInlinedAlready) continue;

    auto root_iter_vars = s->op->root_iter_vars();
    ArrayNode* leaf_vars = s->leaf_iter_vars.CopyOnWrite();
    for (IterVar iv : root_iter_vars) {
      size_t idx = FindNodeRef(leaf_vars, iv);

      /////////////////////////////////////////////////////////////////////////////////////////

      // don't need to rebase path that are binded.
      // if (it != s->iter_var_attrs.end() && (*it).second->bind_thread.defined()) {
      // continue;
      // }

      /////////////////////////////////////////////////////////////////////////////////////////

      if (idx < leaf_vars->data.size()) {
        // insert rebase
        IterVar rebased = IterVarNode::make(Range(), iv->var.copy_with_suffix(".r"), iv->iter_type);
        s->relations.push_back(RebaseNode::make(iv, rebased));

        // Create dimensions
        CHECK(s->leaf_var_dim_map.count(iv)) << iv << " " << s;
        Dimension iv_dim = s->leaf_var_dim_map.at(iv);
        s->leaf_var_dim_map.Set(rebased, Dimension::get_or_create_dimension(
                                             {DimKey::kRebase, iv_dim.operator->(), nullptr}));

        if (s->iter_var_attrs.count(iv)) {
          s->iter_var_attrs.Set(rebased, s->iter_var_attrs.at(iv));

          auto attrs = s->iter_var_attrs.at(iv);
          if (attrs->bind_thread.defined()) {
            // If this is bound IterVar, we need to unbind the parent IterVar, so that only the
            // rebased IterVar is bound at the end.
            s.unbind(iv);
          }
        }
        leaf_vars->data[idx] = rebased;
        rebase_map[iv] = rebased;
      }
    }
  }
  // remap the parent relation
  for (Stage s : sch->stages) {
    if (s->attach_type != kScope && s->attach_type != kSingleKernelScope &&
        s->attach_type != kConditionalThen && s->attach_type != kConditionalElse)
      continue;
    if (rebase_map.count(s->attach_ivar)) {
      s->attach_ivar = rebase_map.at(s->attach_ivar);
    }
  }
  for (Stage s : sch->groups) {
    if (s->attach_type != kScope && s->attach_type != kSingleKernelScope &&
        s->attach_type != kConditionalThen && s->attach_type != kConditionalElse)
      continue;
    if (rebase_map.count(s->attach_ivar)) {
      s->attach_ivar = rebase_map.at(s->attach_ivar);
    }
  }
}

void InjectInline(ScheduleNode* sch) {
  sch->InvalidateCache();

  std::vector<Array<PrimExpr>> new_body(sch->stages.size());
  std::vector<bool> changed(sch->stages.size(), false);
  std::vector<Stmt> new_hybrid_body(sch->stages.size());
  std::vector<bool> hybrid_changed(sch->stages.size(), false);
  // inline all the ops
  for (size_t i = sch->stages.size(); i != 0; --i) {
    Stage stage = sch->stages[i - 1];
    bool print = false;  //(stage->op->name == "lnext_c.ila" || stage->op->name == "lc_prev.ila");

    if (stage->attach_type == kInline) {
      if (print) std::cout << "[INL] Inlining op " << stage->op << std::endl;
      stage->attach_type = kInlinedAlready;
      Array<Var> args;
      PrimExpr body;
      {
        // setup args
        const ComputeOpNode* compute = stage->op.as<ComputeOpNode>();
        CHECK(compute) << "can only inline compute op";
        for (auto dim : compute->root_index_dimensions) {
          args.push_back(compute->GetIterVarFromDim(0, dim)->var);
        }
        CHECK_EQ(compute->body.size(), 1U) << "can only inline compute op with 1 output";
        body = compute->body[0];
        if (print) std::cout << "[INL]   Body " << body << std::endl;
      }
      for (size_t j = i; j < sch->stages.size(); ++j) {
        Stage s = sch->stages[j];
        const ComputeOpNode* compute = s->op.as<ComputeOpNode>();
        const HybridOpNode* hybrid = s->op.as<HybridOpNode>();
        if (compute) {
          auto inlined = stage->op.as<ComputeOpNode>();
          Map<Var, PrimExpr> vmap;
          {
            for (const auto& di : inlined->all_dimensions) {
              Dimension dim = di->dim;
              CHECK(dim->isLoopDim());
              if (compute->dim2var_maps[0].count(dim.as<DimensionNode>())) {
                vmap.Set(di->iv->var, compute->GetIterVarFromDim(0, dim)->var);
              }
            }
          }

          if (!new_body[j].size()) {
            new_body[j] = compute->body;
          }
          PrimExpr old_body = new_body[j][0];
          bool this_changed = false;
          if (new_body[j][0]->IsInstance<tir::ReduceNode>()) {
            // specially handle reduction inline for multiple reductions.
            const tir::ReduceNode* reduce = new_body[j][0].as<tir::ReduceNode>();
            for (size_t k = 1; k < new_body[j].size(); ++k) {
              const tir::ReduceNode* reduce_ = new_body[j][k].as<tir::ReduceNode>();
              CHECK(reduce_);
              CHECK(ReduceEqual(reduce_, reduce)) << "The Reduce inputs of ComputeOp should "
                                                  << "have the same attribute except value_index";
            }
            PrimExpr new_value =
                tir::Inline(tir::EvaluateNode::make(new_body[j][0]), stage->op, args, body)
                    .as<tir::EvaluateNode>()
                    ->value;
            if (!new_value.same_as(new_body[j][0])) {
              this_changed = true;
              changed[j] = true;
              const tir::ReduceNode* r = new_value.as<tir::ReduceNode>();
              CHECK_EQ(new_body[j].size(), r->source.size());
              CHECK(r != nullptr);
              for (size_t k = 0; k < new_body[j].size(); ++k) {
                auto n = make_object<tir::ReduceNode>(*r);
                n->value_index = static_cast<int>(k);
                n->dtype = r->source[k].dtype();
                new_body[j].Set(k, PrimExpr(n));
              }
            }
          } else {
            for (size_t k = 0; k < new_body[j].size(); ++k) {
              PrimExpr new_value =
                  tir::Inline(tir::EvaluateNode::make(new_body[j][k]), stage->op, args, body, vmap)
                      .as<tir::EvaluateNode>()
                      ->value;
              if (!new_value.same_as(new_body[j][k])) {
                new_body[j].Set(k, new_value);
                changed[j] = true;
                this_changed = true;
              }
            }
          }

          if (print && this_changed) {
            std::cout << "[INL]   Inlining in  " << s->op << std::endl;
            std::cout << "[INL]   Before inlining  " << old_body << std::endl;
            std::cout << "[INL]   After inlining  " << new_body[j] << std::endl;
          }

          if (this_changed) {
            for (const auto& di : inlined->all_dimensions) {
              Dimension dim = di->dim;
              CHECK(!dim->isFunDim());
            }
          }
        } else if (hybrid) {
          if (!new_hybrid_body[j].defined()) {
            new_hybrid_body[j] = hybrid->body;
          }
          Stmt new_stmt = tir::Inline(new_hybrid_body[j], stage->op, args, body);
          if (!new_stmt.same_as(new_hybrid_body[j])) {
            new_hybrid_body[j] = new_stmt;
            hybrid_changed[j] = true;
          }
        }
      }
    }
  }
  std::unordered_map<Tensor, Tensor> repl;
  // rewrite dataflow
  for (size_t i = 0; i < sch->stages.size(); ++i) {
    Stage s = sch->stages[i];

    if (s->attach_type == kInlinedAlready) continue;
    if (new_body[i].size()) {
      // Logics from ReplaceDataFlow
      const ComputeOpNode* compute = sch->stages[i]->op.as<ComputeOpNode>();
      CHECK(compute);
      Operation op = s->op;
      if (changed[i]) {
        op = ComputeOpNode::make(compute->name, compute->tag, compute->attrs, compute->axis,
                                 compute->root_index_dimensions, compute->output_shape_storage,
                                 compute->storage_layouts, compute->loop_layout_object, new_body[i],
                                 compute->pred);
      }
      op = op->ReplaceInputs(op, repl);
      if (!op.same_as(s->op)) {
        for (int idx = 0; idx < s->op->num_outputs(); ++idx) {
          repl[s->op.output(idx)] = op.output(idx);
        }
        s->op = op;
      }
    } else if (hybrid_changed[i]) {
      const HybridOpNode* hybrid = sch->stages[i]->op.as<HybridOpNode>();
      CHECK(hybrid);
      Operation op = HybridOpNode::make(hybrid->name, hybrid->tag, hybrid->attrs, hybrid->inputs,
                                        hybrid->outputs, new_hybrid_body[i]);
      op = op->ReplaceInputs(op, repl);
      for (int idx = 0; idx < s->op->num_outputs(); ++idx) {
        repl[s->op.output(idx)] = op.output(idx);
      }
      s->op = op;
    } else {
      Operation op = s->op->ReplaceInputs(s->op, repl);
      if (!op.same_as(s->op)) {
        for (int j = 0; j < op->num_outputs(); ++j) {
          repl[s->op.output(j)] = op.output(j);
        }
        s->op = op;
      }
    }
  }
}

Schedule Schedule::normalize() {
  Schedule sn = copy();
  CheckSchedule(sn, "schedule_dataflow_rewrite.cc:828");
  InjectInline(sn.operator->());
  CheckSchedule(sn, "schedule_dataflow_rewrite.cc:830");
  RebaseNonZeroMinLoop(sn);
  CheckSchedule(sn, "schedule_dataflow_rewrite.cc:832");
  return sn;
}
// Reduction along the factored axis is moved to a new stage. So in
// the original stage, after the rfactor transform, the factored axis
// is no longer a reduction axis, allowing one to parallelize along
// that axis

// Handle reduction factor.
Array<Tensor> Schedule::rfactor(const Tensor& tensor, const IterVar& axis, int factor_axis,
                                Dimension rfactor_dim) {
  bool print = false;
  if (print) std::cout << "[RFACTOR]" << std::endl;
  (*this)->InvalidateCache();
  using tir::ReduceNode;
  CHECK_EQ(axis->iter_type, kCommReduce) << "Can only factor reduction axis";
  Stage reduce_stage = operator[](tensor->op);
  const ComputeOpNode* compute_op = reduce_stage->op.as<ComputeOpNode>();

  CHECK(compute_op) << "Can only factor ComputeOp";
  ArrayNode* leaf_vars = reduce_stage->leaf_iter_vars.CopyOnWrite();
  {
    size_t axis_pos = FindNodeRef(leaf_vars, axis);
    CHECK_NE(axis_pos, leaf_vars->data.size())
        << "Cannot find IterVar " << axis << " in leaf iter vars";
  }
  // Find touched reduction axis.
  std::unordered_map<IterVar, int> touch_map;
  touch_map[axis] = 1;
  te::PassUpBitMaskOr(reduce_stage, &touch_map, true);
  te::PassDownBitMaskOr(reduce_stage, &touch_map, true);
  // skip reduction iteration.
  std::unordered_set<IterVar> skip_bound_check;
  // Verify normal axis are not touched.
  for (IterVar iv : compute_op->axis) {
    CHECK(!touch_map.count(iv)) << "Factor axis touches normal axis.";
    skip_bound_check.insert(iv);
  }
  // get analyzer.
  arith::Analyzer analyzer;
  // Get the replace index
  std::unordered_map<IterVar, Range> dom_map;
  std::unordered_map<IterVar, PrimExpr> value_map;
  for (IterVar iv : compute_op->reduce_axis) {
    if (touch_map.count(iv)) {
      dom_map[iv] = iv->dom;
    } else {
      skip_bound_check.insert(iv);
    }
    analyzer.Bind(iv->var, iv->dom);
  }
  te::PassDownDomain(reduce_stage, &dom_map, &analyzer, true);
  for (IterVar iv : reduce_stage->leaf_iter_vars) {
    if (touch_map.count(iv)) {
      Range dom = dom_map.at(iv);
      if (is_one(dom->extent)) {
        value_map[iv] = dom->min;
      } else {
        value_map[iv] = iv->var;
      }
    }
  }
  te::PassUpIndex(reduce_stage, dom_map, &value_map, true);
  std::vector<PrimExpr> predicates =
      MakeBoundCheck(reduce_stage, dom_map, {}, {}, {}, value_map, true, skip_bound_check, {}, {});

  // Get the factored op node.
  const int factor_axis_pos =
      factor_axis >= 0 ? factor_axis : static_cast<int>(compute_op->axis.size() + 1) + factor_axis;
  CHECK_LE(factor_axis_pos, compute_op->axis.size());
  auto n = make_object<ComputeOpNode>();
  n->name = compute_op->name + ".rf";
  n->output_buffer = compute_op->output_buffer;
  n->output_buffer_dims = compute_op->output_buffer_dims;
  std::unordered_map<const VarNode*, PrimExpr> index_var_sub;
  Dimension new_dim;
  Dimension new_reduction_dim;
  if (rfactor_dim.defined()) {
    new_dim = rfactor_dim;
    CHECK(false);
  } else {
    // new_dim = DimensionNode::make("rfactor", DimensionNode::kRangeDim);
    new_dim = reduce_stage->leaf_var_dim_map.at(axis);
    for (size_t i = reduce_stage->relations.size(); i != 0; --i) {
      if (const SplitNode* r = reduce_stage->relations[i - 1].as<SplitNode>()) {
        if (axis == r->outer) {
          new_reduction_dim = reduce_stage->leaf_var_dim_map.at(r->inner);
          break;
        } else if (axis == r->inner) {
          new_reduction_dim = reduce_stage->leaf_var_dim_map.at(r->outer);
          break;
        }
      }
    }
  }
  std::unordered_map<const VarNode*, PrimExpr> axis_vsub_map;
  if (print)
    std::cout << "[RF] Original op root dims: " << compute_op->root_index_dimensions.size()
              << std::endl;
  {
    // axis relacement.
    IterVar factor_pos_iv = NullValue<IterVar>();
    auto create_factor_pos_iv = [](VarReplacer& replacer, const Range& r, const Var& var) {
      auto iv_node = make_object<IterVarNode>();
      iv_node->dom = replacer.replace(r);
      CHECK(is_zero(iv_node->dom->min)) << "Can only factor reduction domain starting from 0";
      iv_node->var = var;
      iv_node->iter_type = kDataPar;
      return IterVar(iv_node);
    };
    // TODO(ppf): Choose a derived name for the new dimension

    CHECK(factor_axis_pos <= compute_op->axis.size()) << compute_op->axis.size();
    size_t i0 = 0;
    for (; i0 < compute_op->axis.size(); ++i0) {
      auto c_iv = compute_op->axis[i0];
      auto c_dim = compute_op->root_index_dimensions[i0];
      CHECK(c_dim->isLoopDim());
      VarReplacer replacer(axis_vsub_map);
      if (factor_axis_pos == static_cast<int>(i0)) {
        factor_pos_iv = create_factor_pos_iv(replacer, dom_map.at(axis), axis->var);
        n->axis.push_back(factor_pos_iv);
        if (print) std::cout << "[RF] NewOp Axis0 " << factor_pos_iv << std::endl;
        n->all_dimensions.push_back(DimInfoNode::make(new_dim, factor_pos_iv));
      }
      auto new_iv = IterVarNode::make(
          Range::make_by_min_extent(replacer(c_iv->dom->min), replacer(c_iv->dom->extent)),
          Var("iv" + std::to_string(i0), DataType::Int(32)), c_iv->iter_type, c_iv->thread_tag);

      n->axis.push_back(new_iv);
      if (print) std::cout << "[RF] NewOp Axis1 " << new_iv << std::endl;
      n->all_dimensions.push_back(DimInfoNode::make(c_dim, new_iv));
      axis_vsub_map[c_iv->var.as<VarNode>()] = new_iv->var;
    }
    if (factor_axis_pos == static_cast<int>(i0)) {
      VarReplacer replacer(axis_vsub_map);
      factor_pos_iv = create_factor_pos_iv(replacer, dom_map.at(axis), axis->var);
      n->axis.push_back(factor_pos_iv);
      if (print) std::cout << "[RF] NewOp Axis2 " << factor_pos_iv << std::endl;
      n->all_dimensions.push_back(DimInfoNode::make(new_dim, factor_pos_iv));
    }

    CHECK(factor_pos_iv.defined());

    size_t i1 = 0;
    for (; i1 < compute_op->root_index_dimensions.size(); ++i1) {
      if (factor_axis == static_cast<int>(i1)) {
        if (print) std::cout << "[RF] Shape 1 " << new_dim << std::endl;
        n->output_shape_storage.push_back(factor_pos_iv->dom->extent);
        n->root_index_dimensions.push_back(new_dim);
      }

      if (print) std::cout << "[RF] Shape 2 " << compute_op->output_shape_storage[i1] << std::endl;
      n->output_shape_storage.push_back(compute_op->output_shape_storage[i1]);
      n->root_index_dimensions.push_back(compute_op->root_index_dimensions[i1]);
    }
    if (factor_axis == static_cast<int>(i1)) {
      if (print) std::cout << "[RF] Shape 1 " << new_dim << std::endl;
      n->output_shape_storage.push_back(factor_pos_iv->dom->extent);
      n->root_index_dimensions.push_back(new_dim);
    }

    std::unordered_set<const Object*> factor_op_root_vars;
    for (auto iv : n->axis) {
      factor_op_root_vars.insert(iv->var.get());
    }
    VarCollector collector;
    for (auto var_needed : collector.collect(factor_pos_iv->dom)) {
      CHECK(factor_op_root_vars.count(var_needed))
          << "IterVar verification failed during rfactor. Did you specify the correct factor_axis "
             "pos?";
    }

    n->loop_layout_object =
        ModesNode::make_loop_layout(n->root_index_dimensions, n->output_shape_storage, {}, {});
  }
  // predicate generation, copy not touched axis.
  int idx = tensor->value_index;
  const ReduceNode* reduce = compute_op->body[idx].as<ReduceNode>();
  CHECK(reduce) << "Can only rfactor non-inline reductions";
  predicates.push_back(reduce->condition);
  PrimExpr predicate = likely(arith::ComputeReduce<tir::AndNode>(predicates, PrimExpr()));

  std::unordered_map<const VarNode*, PrimExpr> vsub;

  for (size_t i = 0; i < compute_op->reduce_axis.size(); ++i) {
    auto iv = compute_op->reduce_axis[i];
    if (!touch_map.count(iv)) {
      n->reduce_axis.push_back(iv);
      if (compute_op->reduction_dimensions.size() > 0) {
        n->reduction_dimensions.push_back(compute_op->reduction_dimensions[i]);
      }
      if (print) std::cout << "[RF] NewOp Raxs1 " << iv << std::endl;
    } else {
      CHECK(value_map.count(iv));
      PrimExpr index = value_map.at(iv);
      vsub[iv->var.get()] = index;
    }
  }

  // Copy touched axis.
  for (IterVar iv : reduce_stage->leaf_iter_vars) {
    if (touch_map.count(iv) && !iv.same_as(axis)) {
      CHECK_EQ(iv->iter_type, kCommReduce);
      auto ncpy = make_object<IterVarNode>(*iv.operator->());
      ncpy->dom = dom_map.at(iv);
      auto new_iv = IterVar(ncpy);
      n->reduce_axis.push_back(new_iv);
      if (print) std::cout << "[RF] NewOp Raxs2 " << new_iv << std::endl;
      if (compute_op->reduction_dimensions.size() > 0) {
        auto dim = DimensionNode::make("rf_reduction_dim", DimensionNode::DimensionType::kRangeDim);
        n->reduction_dimensions.push_back(dim);
        n->all_dimensions.push_back(DimInfoNode::make(dim, new_iv));
        if (print) std::cout << "[RF]   Raxs2 has dim" << std::endl;
      }
    }
  }
  VarReplacer replacer(vsub);
  Array<PrimExpr> new_source =
      tir::UpdateArray(reduce->source, [&replacer](const PrimExpr& e) { return replacer(e); });

  PrimExpr new_pred = replacer(predicate);

  std::vector<PrimExpr> body;
  for (size_t idx = 0; idx < reduce->source.size(); ++idx) {
    // Substitute old index variables with the new ones
    auto unreplaced_body = ReduceNode::make(reduce->combiner, new_source, n->reduce_axis, new_pred,
                                            idx, n->reduction_dimensions);
    body.emplace_back(VarReplacer(index_var_sub)(VarReplacer(axis_vsub_map)(unreplaced_body)));
  }
  n->body = Array<PrimExpr>(body);
  std::vector<PrimExpr> pred;
  for (const auto& p : compute_op->pred) {
    // Substitute old index variables with the new ones
    pred.emplace_back(VarReplacer(index_var_sub)(VarReplacer(axis_vsub_map)(p)));
  }
  n->pred = Array<PrimExpr>(pred);

  // if (print) std::cout << n->body << std::endl;
  // refresh relations, keep the un-touched relations.
  Array<IterVarRelation> rels;
  for (IterVarRelation rel : reduce_stage->relations) {
    bool touched = false;
    if (const SplitNode* r = rel.as<SplitNode>()) {
      if (touch_map.count(r->parent)) touched = true;
    } else if (const FuseNode* r = rel.as<FuseNode>()) {
      if (touch_map.count(r->fused)) touched = true;
    } else if (const RebaseNode* r = rel.as<RebaseNode>()) {
      if (touch_map.count(r->parent)) touched = true;
    } else {
      LOG(FATAL) << "unknown relation type";
    }
    if (!touched) {
      rels.push_back(rel);
    }
  }

  if (print)
    std::cout << "[RF] Factor op root dims: " << n->root_index_dimensions.size() << std::endl;
  // initialize the factored stage.
  n->RefreshDimVarMappings();
  Operation factor_op(n);
  ArrayNode* stages = (*this)->stages.CopyOnWrite();
  size_t stage_pos = FindNodeRef(stages, reduce_stage);
  Stage factor_stage = Stage(factor_op);
  factor_stage->relations = rels;
  CHECK_LT(stage_pos, stages->data.size());
  stages->data.insert(stages->data.begin() + stage_pos, factor_stage);
  (*this)->stage_map.Set(factor_op, factor_stage);
  factor_stage->group = reduce_stage->group;
  if (factor_stage->group.defined()) {
    ++factor_stage->group->num_child_stages;
  }

  ///////////////////////////////////////////////////////////////////////////////////////////
  // Replace the old reduction.
  IterVar repl_red_axis = NullValue<IterVar>();
  Array<IterVar> new_axis;
  Array<PrimExpr> preds;

  {
    std::unordered_map<const VarNode*, PrimExpr> vsub;
    for (const auto& iv : compute_op->axis) {
      VarReplacer var_replacer(vsub);
      IterVar new_iv = IterVarNode::make(
          Range::make_by_min_extent(var_replacer(iv->dom->min), var_replacer(iv->dom->extent)),
          iv->var.copy_with_suffix(".rf"), iv->iter_type, iv->thread_tag);
      new_axis.push_back(new_iv);

      vsub[iv->var.as<VarNode>()] = new_iv->var;
    }

    VarReplacer replacer(vsub);
    for (const auto& p : compute_op->pred) {
      preds.push_back(replacer(p));
    }
    repl_red_axis = reduce_axis(replacer.replace(dom_map.at(axis)), axis->var->name_hint + ".v");
  }
  ///////////////////////////////////////////////////////////////////////////////////////////

  // These are the newly introduced tensors that store he partial sums
  Array<Tensor> factor_tensors;
  Array<Tensor> old_tensors;
  int size = factor_op->num_outputs();
  for (int idx = 0; idx < size; ++idx) {
    factor_tensors.push_back(factor_op.output(idx));
    old_tensors.push_back(reduce_stage->op.output(idx));
  }

  if (print) std::cout << "[RF]   New Dim " << new_dim << std::endl;
  auto body_lambda = [&](const Map<Dimension, Var>& args) {
    Array<PrimExpr> indices;
    for (auto dim : n->root_index_dimensions) {
      if (print) std::cout << "[RF]   Index Dim " << dim << std::endl;
      if (dim == new_dim) {
        indices.push_back(repl_red_axis);
      } else {
        CHECK(args.count(dim)) << "Dim " << dim->name << " not in args";
        indices.push_back(args.at(dim));
      }
    }

    CHECK(indices.size() > 0) << " " << n->root_index_dimensions.size();

    Array<PrimExpr> factor_exprs;
    for (int idx = 0; idx < size; ++idx) {
      auto expr = factor_tensors[idx](indices);
      // if (print) std::cout << "[RF] Body factor expr " << expr << std::endl;
      factor_exprs.push_back(expr);
    }
    Array<PrimExpr> reductions;
    Array<IterVar> axis = {repl_red_axis};
    PrimExpr cond = const_true();
    for (int idx = 0; idx < size; ++idx) {
      reductions.push_back(
          ReduceNode::make(reduce->combiner, factor_exprs, axis, cond, idx, {new_dim}));
    }
    return reductions;
  };

  // if (print) std::cout << "[RF] " << preds[0] << std::endl;
  auto pred_lambda = [&](const Map<Dimension, Var>& args) { return preds; };

  // The tensors corresponding to the original stage
  // if (print) std::cout << "[RF] Old tensors " << old_tensors.size() << std::endl;
  Array<Tensor> repl_tensors =
      compute(old_tensors[0]->shape, body_lambda, pred_lambda, reduce_stage->op->name + ".repl", "",
              Map<std::string, ObjectRef>(), new_axis, compute_op->root_index_dimensions);

  std::unordered_map<Tensor, Tensor> vmap;
  std::unordered_map<Tensor, Tensor> rvmap;
  for (int idx = 0; idx < size; ++idx) {
    vmap[old_tensors[idx]] = repl_tensors[idx];
    rvmap[repl_tensors[idx]] = old_tensors[idx];
  }
  ReplaceDataFlow((*this)->stages, (*this)->cacheTensorInfos, &vmap, &rvmap);
  // revamp the reduction stage.
  reduce_stage->op = repl_tensors[0]->op;
  reduce_stage->all_iter_vars = repl_tensors[0]->op->root_iter_vars();
  reduce_stage->leaf_iter_vars = reduce_stage->all_iter_vars;
  reduce_stage->relations = Array<IterVarRelation>();
  // reduce_stage->leaf_var_dim_map.clear();
  for (auto iv : reduce_stage->leaf_iter_vars) {
    reduce_stage->leaf_var_dim_map.Set(
        iv, repl_tensors[0]->op.as<ComputeOpNode>()->GetDimensionFromVar(0, iv->var));
  }

  CheckSchedule(*this, "schedule_dataflow_rewrite.cc:748_end_" + tensor->op->name);
  return factor_tensors;
}
}  // namespace te
}  // namespace tvm
