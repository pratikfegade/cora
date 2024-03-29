/*
 * Licensed the Apache Software Foundation (ASF) under one
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
 * \file tvm/te/operation.h
 * \brief Operation node can generate one or multiple Tensors
 */
#ifndef TVM_TE_OPERATION_H_
#define TVM_TE_OPERATION_H_

#include <tvm/arith/analyzer.h>
#include <tvm/ir/attrs.h>
#include <tvm/te/cache_info.h>
#include <tvm/te/dimension.h>
#include <tvm/te/dimension_relations.h>
#include <tvm/te/schedule.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/uninterp_fun.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace tvm {
/*! \brief Tensor expression language DSL. */
namespace te {

/*!
 * \brief Temporary data structure to store union
 *  of bounds of each axis of Tensor.
 */
struct TensorDom {
  // constructor
  explicit TensorDom(int ndim) : data(ndim) {}
  /*! \brief The domain data */
  std::vector<std::vector<IntSet>> data;
  /*! \brief Used only when the consumer is a scan, specify the values
      for the scan axis, which is a loop variable, as opposed to the
      other dimensions in data, which are index variables. */
  // std::vector<std::vector<IntSet> > scan_axis_data;
};

struct DimVarEntry {
  Dimension dim;
  IterVar iv;
};

class DimInfo;

class DimInfoNode : public runtime::Object {
 public:
  Dimension dim;
  IterVar iv;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("dim", &dim);
    v->Visit("iv", &iv);
  }

  TVM_DLL static DimInfo make(Dimension dim, IterVar iv);

  static constexpr const char* _type_key = "te.DimInfo";
  TVM_DECLARE_FINAL_OBJECT_INFO(DimInfoNode, Object);
};

class DimInfo : public runtime::ObjectRef {
 public:
  DimInfo() {}
  // construct from shared ptr.
  explicit DimInfo(runtime::ObjectPtr<runtime::Object> n) : ObjectRef(n) {}
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const DimInfoNode* operator->() const;

  /*! \brief specify container node */
  using ContainerType = DimInfoNode;
};

inline const DimInfoNode* DimInfo::operator->() const {
  return static_cast<const DimInfoNode*>(data_.get());
}

/*!
 * \brief Base class of all operation nodes
 */
class OperationNode : public tir::FunctionBaseNode {
 public:
  /*! \brief optional name of the operation */
  std::string name;
  /*! \brief optional tag of the operation */
  std::string tag;
  /*! \brief additional attributes of the operation*/
  Map<std::string, ObjectRef> attrs;
  /*! \return name of the operation */
  const std::string& func_name() const final { return name; }
  /*!
   * \return The list of iteration variable at root
   * \note root_iter_vars decides the shape of the outputs.
   */
  virtual Array<IterVar> root_iter_vars() const = 0;
  /*!
   * \brief Get data type. i-th output tensor.
   * \param i The output index.
   * \return type of i-th output.
   */
  virtual DataType output_dtype(size_t i) const = 0;
  /*!
   * \brief Get shape of i-th output tensor.
   * \param i The output index.
   * \return shape of i-th output.
   */
  virtual Array<PrimExpr> output_shape(size_t i) const = 0;
  /*!
   * \brief Get the optional layout of i-th output tensor.
   * \param i The output index.
   * \return the layout of i-th output.
   */
  virtual Modes output_layout(size_t i) const { return NullValue<Modes>(); };
  /*!
   * \brief Get the optional layout representing the loop nest.
   * \return the layout representing the loop nest.
   */
  virtual Modes loop_layout() const { return NullValue<Modes>(); };
  /*!
   * \brief List all the input Tensors.
   * \return List of input tensors.
   */
  virtual Array<Tensor> InputTensors() const = 0;
  /*!
   * \brief List all the input Tensors. This also includes tensors
   * that are not expected to be present in the emitted code. For
   * example, by default, the ScanOp does not include tensors in the
   * UFs for it's spatial axes in the results.
   * \return List of input
   * tensors.
   */
  virtual inline Array<Tensor> InputTensorsWithUnemitted() const { return this->InputTensors(); }
  /*!
   * \brief List only the input tensors in the body of the op. This
   * also includes tensors that are not expected to be present in the
   * emitted code. For example, by default, the ScanOp does not
   * include tensors in the UFs for it's spatial axes in the results.
   * \return List of input tensors.
   */
  virtual inline Array<Tensor> InputTensorsOnlyBody() const { return this->InputTensors(); }
  /*!
   * \brief Replace the input of the operation by pattern specified by rmap.
   *
   * \param self The reference to self.
   * \param rmap The replacement map.
   * \return self if nothing is replaced, otherwise return replaced op.
   */
  virtual Operation ReplaceInputs(const Operation& self,
                                  const std::unordered_map<Tensor, Tensor>& rmap) const = 0;
  /*!
   * \brief Propagate the bounds to inputs
   * \param self The reference to self.
   * \param analyzer The analyzer to be used in the function.
   * \param dom_map the domain map of Variables(corresponds to root_iter_vars)
   * \param out_dom_map The output domain.
   *  The function is only asked to fill the bounds for Tensors that
   *  is already in the out_dom_map
   */
  virtual void PropBoundToInputs(const Operation& self, arith::Analyzer* analyzer,
                                 const std::unordered_map<const VarNode*, IntSet>& dom_map,
                                 std::unordered_map<Tensor, TensorDom>* out_dom_map) const = 0;
  /*!
   * \brief Gather the bound from output tensor.
   *  Set the range of each root_iter_vars in the op to out_dom_map
   *
   * \param self The reference to self.
   * \param tensor_dom Domain map of Tensor->access set of each dimension.
   * \param out_dom_map The output domain map of each IterVar to be setted.
   */
  virtual void GatherBound(const Operation& self,
                           const std::unordered_map<Tensor, TensorDom>& tensor_dom,
                           std::unordered_map<IterVar, Range>* out_dom_map,
                           const Map<FunctionRef, CacheInfo> cacheTensorInfos) const = 0;
  /*!
   * \brief Build the Realize statement that realizes
   *   the op's output tensors.
   * \param stage the op's stage.
   * \param realize_map The realization domain map of the operators.
   * \param body The body that is going to get
   * \return A realization statement that wraps body.
   */
  virtual Stmt BuildRealize(const Stage& stage,
                            const std::unordered_map<IterVar, Range>& realize_map,
                            const Stmt& body) const = 0;
  /*!
   * \brief Build the statement that provide the output tensors.
   * \param stage The schedule stage of the op.
   * \param dom_map The domain map of all iteration domains.
   * \param debug_keep_trivial_loop Whether keep trivial loops with extent of 1
   * \return A statement that add production and wraps consumer.
   */
  virtual Stmt BuildProvide(const Stage& stage, const std::unordered_map<IterVar, Range>& dom_map,
                            const std::unordered_map<std::string, Range>& env_dom_map,
                            const std::unordered_map<std::string, IterVar>& env_var_map,
                            const std::unordered_map<const VarNode*, std::string>& bind_map,
                            const Map<Stage, Array<Stage>>& attach_stages,
                            const Map<Stage, Array<IterVar>>& attach_vars,
                            bool debug_keep_trivial_loop) const = 0;

  static constexpr const char* _type_key = "Operation";

  TVM_DECLARE_BASE_OBJECT_INFO(OperationNode, Object);
};

class TVM_DLL BaseVarDimOpNode : public OperationNode {
 public:
  std::vector<std::unordered_map<const DimensionNode*, DimVarEntry>> dim2var_maps;
  std::unordered_map<const VarNode*, const DimensionNode*> var2dim_map;

  IterVar GetIterVarFromDim(int val_idx, Dimension dim, bool only_loop_dims = false) const;
  Dimension GetDimensionFromVar(int val_idx, Var var) const;

  virtual Dimension GetBaseIndexDimension(size_t val_idx, size_t dim_idx) const = 0;
  virtual Array<DimInfo> GetAllDimensions() const;
  virtual Array<Dimension> GetRootIndexDimensions(size_t val_idx) const = 0;

  static constexpr const char* _type_key = "BaseVarDimOp";
  TVM_DECLARE_BASE_OBJECT_INFO(BaseVarDimOpNode, OperationNode);
};

/*!
 * \brief A placeholder op represents an input placeholder.
 */
class PlaceholderOpNode : public BaseVarDimOpNode {
 public:
  /*! \brief The shape of the input */
  Array<PrimExpr> shape;
  /*! \brief The data type of the input. */
  DataType dtype;
  /*! \brief The named dimensions for indexing the output tensor */
  Array<Dimension> self_index_dimensions;
  /*! \brief DimInfos */
  Array<DimInfo> all_dimensions;
  /*! \brief Storage layout */
  Modes layout;

  Modes output_layout(size_t i) const { return layout; };

  // override behavior.
  int num_outputs() const final;
  Array<IterVar> root_iter_vars() const final;
  DataType output_dtype(size_t i) const final;
  Array<PrimExpr> output_shape(size_t i) const final;
  Array<Tensor> InputTensors() const final;
  Operation ReplaceInputs(const Operation& self,
                          const std::unordered_map<Tensor, Tensor>& rmap) const final;
  void PropBoundToInputs(const Operation& self, arith::Analyzer* analyzer,
                         const std::unordered_map<const VarNode*, IntSet>& dom_map,
                         std::unordered_map<Tensor, TensorDom>* out_dom_map) const final;
  void GatherBound(const Operation& self, const std::unordered_map<Tensor, TensorDom>& tensor_dom,
                   std::unordered_map<IterVar, Range>* out_dom_map,
                   const Map<FunctionRef, CacheInfo> cacheTensorInfos) const final;
  Stmt BuildRealize(const Stage& stage, const std::unordered_map<IterVar, Range>& realize_map,
                    const Stmt& body) const final;
  Stmt BuildProvide(const Stage& stage, const std::unordered_map<IterVar, Range>& dom_map,
                    const std::unordered_map<std::string, Range>& env_dom_map,
                    const std::unordered_map<std::string, IterVar>& env_var_map,
                    const std::unordered_map<const VarNode*, std::string>& bind_map,
                    const Map<Stage, Array<Stage>>& attach_stages,
                    const Map<Stage, Array<IterVar>>& attach_vars,
                    bool debug_keep_trivial_loop) const final;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("tag", &tag);
    v->Visit("attrs", &attrs);
    v->Visit("shape", &shape);
    v->Visit("dtype", &dtype);
    v->Visit("layout", &layout);
  }
  static Operation make(std::string name, Array<PrimExpr> shape, DataType dtype);

  static Operation make(std::string name, Array<PrimExpr> shape, Modes layout, DataType dtype,
                        Array<Dimension> self_index_expressions, Array<Dimension> dimensions,
                        Array<IterVar> itervars, Array<UninterpFun> uninterpfuns);

  Dimension GetBaseIndexDimension(size_t val_idx, size_t dim_idx) const final;
  Array<DimInfo> GetAllDimensions() const final;
  Array<Dimension> GetRootIndexDimensions(size_t val_idx) const final;

  void set_storage_layout(Modes leaf_layout);

  static constexpr const char* _type_key = "PlaceholderOp";
  TVM_DECLARE_FINAL_OBJECT_INFO(PlaceholderOpNode, BaseVarDimOpNode);
};

/*!
 * \brief A Compute op that compute a tensor on certain domain.
 * This is the base class for ComputeOp (operating on a scalar at a time) and
 * TensorComputeOp (operating on a TensorSlice at a time)
 */
class TVM_DLL BaseComputeOpNode : public BaseVarDimOpNode {
 public:
  /*! \brief Is this a RA rec op, or a lowered ILA op */
  bool is_rec_op = false;

  /*! \brief IterVar on each axis */
  Array<IterVar> axis;
  /*! \brief IterVar on each reduction axis, if the body is a Reduce */
  Array<IterVar> reduce_axis;

  /*! \brief Output shape */
  Array<PrimExpr> output_shape_storage;
  /*! \brief Realize bounds */
  Array<Range> realize_bounds;
  std::string who_set_realize_bounds = "No one yet";
  /*! \brief The named dimensions to index the output tensor */
  Array<Dimension> root_index_dimensions;
  /*! \brief The named dimensions corresponding to the reduction axis, if any */
  Array<Dimension> reduction_dimensions;

  Array<DimInfo> all_dimensions;

  // /*! \brief Index variables */
  // Array<IterVar> index_variables;
  // /*! \brief Values of the index variables in terms of the loop
  //     iteration variables */
  // Array<UninterpFun> index_expressions;
  // /*! \brief The named dimensions for indexing tensors */
  // Array<Dimension> index_dimensions;
  // /*! \brief The named dimensions for iterating over the output tensor */
  // Array<Dimension> loop_dimensions;

  /*! \brief If this is a non-global output op, the output can be
      written to this buffer */
  Buffer output_buffer;
  /*! \brief The index dimensions of the buffer */
  Array<Dimension> output_buffer_dims;
  /*! \brief The optional layouts of the output buffers */
  Array<Modes> storage_layouts;
  /*! \brief The optional layouts of the output buffers */
  Modes loop_layout_object;

  Modes output_layout(size_t i) const final;

  Modes loop_layout() const final;

  void set_realize_bounds(Array<Range>, std::string caller);

  void set_storage_layout(int i, Modes leaf_layout);

  void set_all_dimensions(Array<DimInfo>);

  // override functions
  Array<IterVar> root_iter_vars() const final;
  Dimension GetBaseIndexDimension(size_t val_idx, size_t dim_idx) const final;
  Array<PrimExpr> output_shape(size_t idx) const final;
  void GatherBound(const Operation& self, const std::unordered_map<Tensor, TensorDom>& tensor_dom,
                   std::unordered_map<IterVar, Range>* out_dom_map,
                   const Map<FunctionRef, CacheInfo> cacheTensorInfos) const final;

  Region GetRealizeBounds(const Stage& stage,
                          const std::unordered_map<IterVar, Range>& realize_map) const;

  Stmt BuildRealize(const Stage& stage, const std::unordered_map<IterVar, Range>& realize_map,
                    const Stmt& body) const final;
  virtual size_t num_schedulable_dims() const = 0;

  Array<DimInfo> GetAllDimensions() const;
  Array<Dimension> GetRootIndexDimensions(size_t val_idx) const;

  static constexpr const char* _type_key = "BaseComputeOp";
  TVM_DECLARE_BASE_OBJECT_INFO(BaseComputeOpNode, BaseVarDimOpNode);
};

/*!
 * \brief A Compute op that compute a tensor on certain domain.
 */
class TVM_DLL ComputeOpNode : public BaseComputeOpNode {
 public:
  /*! \brief the compute expression */
  Array<PrimExpr> body;
  /*! \brief Predicates under which to compute the body */
  Array<PrimExpr> pred;
  /*! \brief constructor */
  ComputeOpNode() {}
  // override functions
  int num_outputs() const final;
  DataType output_dtype(size_t i) const final;
  Array<Tensor> InputTensors() const final;
  Operation ReplaceInputs(const Operation& self,
                          const std::unordered_map<Tensor, Tensor>& rmap) const final;
  void PropBoundToInputs(const Operation& self, arith::Analyzer* analyzer,
                         const std::unordered_map<const VarNode*, IntSet>& dom_map,
                         std::unordered_map<Tensor, TensorDom>* out_dom_map) const final;
  Stmt BuildProvide(const Stage& stage, const std::unordered_map<IterVar, Range>& dom_map,
                    const std::unordered_map<std::string, Range>& env_dom_map,
                    const std::unordered_map<std::string, IterVar>& env_var_map,
                    const std::unordered_map<const VarNode*, std::string>& bind_map,
                    const Map<Stage, Array<Stage>>& attach_stages,
                    const Map<Stage, Array<IterVar>>& attach_vars,
                    bool debug_keep_trivial_loop) const final;
  size_t num_schedulable_dims() const final;
  Array<Tensor> InputTensorsOnlyBody() const;

  void RefreshDimVarMappings();

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("tag", &tag);
    v->Visit("attrs", &attrs);
    v->Visit("axis", &axis);
    v->Visit("reduce_axis", &reduce_axis);
    v->Visit("body", &body);
    v->Visit("output_buffer", &output_buffer);
    v->Visit("output_buffer_dims", &output_buffer_dims);
    v->Visit("storage_layouts", &storage_layouts);
    v->Visit("loop_layout_object", &loop_layout_object);
  }
  static Operation make(std::string name, std::string tag, Map<std::string, ObjectRef> attrs,
                        Array<IterVar> axis, Array<Dimension> dimensions,
                        Array<PrimExpr> output_shape_storage, Array<Modes> storage_layouts,
                        Modes loop_layout, Array<PrimExpr> body, Array<PrimExpr> pred);

  static Operation make(std::string name, std::string tag, Map<std::string, ObjectRef> attrs,
                        Array<IterVar> axis, Array<PrimExpr> body);

  static constexpr const char* _type_key = "ComputeOp";
  TVM_DECLARE_FINAL_OBJECT_INFO(ComputeOpNode, BaseComputeOpNode);
};

/*!
 * \brief A TenorCompute op that compute a tensor with an tensor intrinsic.
 */
class TensorComputeOpNode : public BaseComputeOpNode {
 public:
  /*! \brief number of axes that can be scheduled */
  int schedulable_ndim;
  /*! \brief TensorIntrin used to compute */
  TensorIntrin intrin;
  /*! \brief input tensors of intrin */
  Array<Tensor> inputs;
  /*! \brief region of input tensors */
  Array<Region> input_regions;
  /*! \brief scalar expression inputs */
  Array<PrimExpr> scalar_inputs;
  /*! \brief constructor */
  TensorComputeOpNode() {}
  // override functions
  int num_outputs() const final;
  DataType output_dtype(size_t i) const final;
  Array<Tensor> InputTensors() const final;
  Operation ReplaceInputs(const Operation& self,
                          const std::unordered_map<Tensor, Tensor>& rmap) const final;
  void PropBoundToInputs(const Operation& self, arith::Analyzer* analyzer,
                         const std::unordered_map<const VarNode*, IntSet>& dom_map,
                         std::unordered_map<Tensor, TensorDom>* out_dom_map) const final;
  Stmt BuildProvide(const Stage& stage, const std::unordered_map<IterVar, Range>& dom_map,
                    const std::unordered_map<std::string, Range>& env_dom_map,
                    const std::unordered_map<std::string, IterVar>& env_var_map,
                    const std::unordered_map<const VarNode*, std::string>& bind_map,
                    const Map<Stage, Array<Stage>>& attach_stages,
                    const Map<Stage, Array<IterVar>>& attach_vars,
                    bool debug_keep_trivial_loop) const final;
  size_t num_schedulable_dims() const final;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("tag", &tag);
    v->Visit("axis", &axis);
    v->Visit("reduce_axis", &reduce_axis);
    v->Visit("schedulable_ndim", &schedulable_ndim);
    v->Visit("intrin", &intrin);
    v->Visit("inputs", &inputs);
    v->Visit("input_regions", &input_regions);
    v->Visit("scalar_inputs", &scalar_inputs);
  }
  static Operation make(std::string name, std::string tag, Array<IterVar> axis,
                        Array<IterVar> reduce_axis, int schedulable_ndim, TensorIntrin intrin,
                        Array<Tensor> tensors, Array<Region> regions,
                        Array<PrimExpr> scalar_inputs);

  static constexpr const char* _type_key = "TensorComputeOp";
  TVM_DECLARE_FINAL_OBJECT_INFO(TensorComputeOpNode, BaseComputeOpNode);
};

/*!
 * \brief Symbolic scan.
 */
class ScanOpNode : public BaseVarDimOpNode {
 public:
  /*! \brief Is this a RA rec op, or a lowered ILA op */
  bool is_rec_op = false;

  /*! \brief IterVar to scan over */
  IterVar scan_axis;
  /*! \brief the initialization tensors */
  Array<Tensor> init;
  /*! \brief the update function represented by tensor */
  Array<Tensor> update;
  /*! \brief The placeholder to refer as states in update. */
  Array<Tensor> state_placeholder;
  /*!
   * \brief the inputs to the scan, these are optionally provided
   *  But they can be helpful to provide hints to speedup get of scan body.
   */
  Array<Tensor> inputs;
  /*!
   * \brief Spatial axis to indicate spatial dimension of each output.
   *  They corresponds to flattened spatial axis of the outputs.
   *
   *  [output[0].axis[1], output[0].axis[2]... output[k].axis[j]...]
   *  These are auxiliary data structure for storing result of bound inference.
   *  They do not corresponds to splittable iterations, thus the name comes
   *  with underscore.
   */
  Array<IterVar> spatial_axis_;
  Dimension scan_dim;

  Array<Dimension> spatial_dimensions_;
  // Loops that this operation will actually generate in the lowered
  // IR
  Array<Dimension> explicit_dims;
  Array<IterVar> explicit_loop_ivs;
  Array<DimInfo> explicit_dimensions;
  // This denotes if there is an explicit init stage for this scan, or
  // if the init stage is folded in as in the case of data structure
  // scans.
  bool init_separate;

  /*! \brief constructor */
  ScanOpNode() {}
  // override behavior.
  int num_outputs() const final;
  Array<IterVar> root_iter_vars() const final;
  DataType output_dtype(size_t i) const final;
  Array<PrimExpr> output_shape(size_t i) const final;
  Dimension GetBaseIndexDimension(size_t val_idx, size_t dim_idx) const final;
  Array<Tensor> InputTensors() const final;
  Array<Tensor> InputTensorsWithUnemitted() const override;
  Operation ReplaceInputs(const Operation& self,
                          const std::unordered_map<Tensor, Tensor>& rmap) const final;
  void PropBoundToInputs(const Operation& self, arith::Analyzer* analyzer,
                         const std::unordered_map<const VarNode*, IntSet>& dom_map,
                         std::unordered_map<Tensor, TensorDom>* out_dom_map) const final;
  void GatherBound(const Operation& self, const std::unordered_map<Tensor, TensorDom>& tensor_dom,
                   std::unordered_map<IterVar, Range>* out_dom_map,
                   const Map<FunctionRef, CacheInfo> cacheTensorInfos) const final;
  Stmt BuildRealize(const Stage& stage, const std::unordered_map<IterVar, Range>& realize_map,
                    const Stmt& body) const final;
  Stmt BuildProvide(const Stage& stage, const std::unordered_map<IterVar, Range>& dom_map,
                    const std::unordered_map<std::string, Range>& env_dom_map,
                    const std::unordered_map<std::string, IterVar>& env_var_map,
                    const std::unordered_map<const VarNode*, std::string>& bind_map,
                    const Map<Stage, Array<Stage>>& attach_stages,
                    const Map<Stage, Array<IterVar>>& attach_vars,
                    bool debug_keep_trivial_loop) const final;
  Array<Dimension> GetRootIndexDimensions(size_t val_idx) const;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("tag", &tag);
    v->Visit("attrs", &attrs);
    v->Visit("scan_axis", &scan_axis);
    v->Visit("init", &init);
    v->Visit("update", &update);
    v->Visit("state_placeholder", &state_placeholder);
    v->Visit("inputs", &inputs);
    v->Visit("spatial_axis_", &spatial_axis_);
  }
  static Operation make(std::string name, std::string tag, Map<std::string, ObjectRef> attrs,
                        UninterpFun range_min_uf, UninterpFun range_max_uf, Dimension scan_dim,
                        bool init_separate, Array<Tensor> init, Array<Tensor> update,
                        Array<Tensor> state_placeholder, Array<Tensor> input,
                        Array<Dimension> explicit_loops, Array<UninterpFun> explicit_min_ufs,
                        Array<UninterpFun> explicit_extent_ufs);

  IterVar RefreshDimVarMappings(UninterpFun range_min_uf, UninterpFun range_max_uf,
                                Array<Dimension> explicit_loops,
                                Array<UninterpFun> explicit_min_ufs,
                                Array<UninterpFun> explicit_max_ufs);

  static constexpr const char* _type_key = "ScanOp";
  TVM_DECLARE_FINAL_OBJECT_INFO(ScanOpNode, BaseVarDimOpNode);

 private:
  Array<Tensor> InputTensors(bool includeAll) const;
};

/*!
 * \brief Symbolic scan.
 */
class ConditionalOpNode : public BaseVarDimOpNode {
 public:
  /*! \brief the initialization tensors */
  Array<Tensor> from_then;
  Array<Tensor> then_case;
  /*! \brief the update function represented by tensor */
  Array<Tensor> from_else;
  Array<Tensor> else_case;
  /*! \brief theif-condition */
  PrimExpr condition;

  Array<IterVar> spatial_axis_;
  Array<Dimension> spatial_dimensions_;
  // Loops that this operation will actually generate in the lowered
  // IR
  Array<Dimension> explicit_dims;
  Array<IterVar> explicit_loop_ivs;
  Array<DimInfo> explicit_dimensions;

  /*! \brief constructor */
  ConditionalOpNode() {}
  // override behavior.
  int num_outputs() const final;
  Array<IterVar> root_iter_vars() const final;
  DataType output_dtype(size_t i) const final;
  Array<PrimExpr> output_shape(size_t i) const final;
  Dimension GetBaseIndexDimension(size_t val_idx, size_t dim_idx) const final;
  Array<Tensor> InputTensors() const final;
  Array<Tensor> InputTensorsWithUnemitted() const override;
  Operation ReplaceInputs(const Operation& self,
                          const std::unordered_map<Tensor, Tensor>& rmap) const final;
  void PropBoundToInputs(const Operation& self, arith::Analyzer* analyzer,
                         const std::unordered_map<const VarNode*, IntSet>& dom_map,
                         std::unordered_map<Tensor, TensorDom>* out_dom_map) const final;
  void GatherBound(const Operation& self, const std::unordered_map<Tensor, TensorDom>& tensor_dom,
                   std::unordered_map<IterVar, Range>* out_dom_map,
                   const Map<FunctionRef, CacheInfo> cacheTensorInfos) const final;
  Stmt BuildRealize(const Stage& stage, const std::unordered_map<IterVar, Range>& realize_map,
                    const Stmt& body) const final;
  Stmt BuildProvide(const Stage& stage, const std::unordered_map<IterVar, Range>& dom_map,
                    const std::unordered_map<std::string, Range>& env_dom_map,
                    const std::unordered_map<std::string, IterVar>& env_var_map,
                    const std::unordered_map<const VarNode*, std::string>& bind_map,
                    const Map<Stage, Array<Stage>>& attach_stages,
                    const Map<Stage, Array<IterVar>>& attach_vars,
                    bool debug_keep_trivial_loop) const final;
  Array<Dimension> GetRootIndexDimensions(size_t val_idx) const;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("tag", &tag);
    v->Visit("attrs", &attrs);
    v->Visit("then_case", &then_case);
    v->Visit("else_case", &else_case);
    v->Visit("condition", &condition);
    v->Visit("spatial_dimensions_", &spatial_dimensions_);
    v->Visit("explicit_dims", &explicit_dims);
    v->Visit("explicit_loop_ivs", &explicit_loop_ivs);
    v->Visit("explicit_dimensions", &explicit_dimensions);
  }
  static Operation make(std::string name, std::string tag, Map<std::string, ObjectRef> attrs,
                        UninterpFun condition_uf, Array<Tensor> from_then, Array<Tensor> then_case,
                        Array<Tensor> from_else, Array<Tensor> else_case,
                        Array<Dimension> explicit_loops, Array<UninterpFun> explicit_min_ufs,
                        Array<UninterpFun> explicit_extent_ufs);

  static constexpr const char* _type_key = "ConditionalOp";
  TVM_DECLARE_FINAL_OBJECT_INFO(ConditionalOpNode, BaseVarDimOpNode);

 private:
  Array<Tensor> InputTensors(bool includeAll) const;
};

/*!
 * \brief Symbolic scan.
 */
class SpecializationEnvelopeOpNode : public BaseVarDimOpNode {
 public:
  Array<Array<Tensor>> inputs;
  std::vector<const BaseVarDimOpNode*> input_ops;
  Array<Dimension> spatial_dimensions_;

  /*! \brief constructor */
  SpecializationEnvelopeOpNode() {}
  // override behavior.
  int num_outputs() const final;
  Array<IterVar> root_iter_vars() const final;
  DataType output_dtype(size_t i) const final;
  Array<PrimExpr> output_shape(size_t i) const final;
  Dimension GetBaseIndexDimension(size_t val_idx, size_t dim_idx) const final;
  Array<Tensor> InputTensors() const final;
  Array<Tensor> InputTensorsWithUnemitted() const override;
  Operation ReplaceInputs(const Operation& self,
                          const std::unordered_map<Tensor, Tensor>& rmap) const final;
  void PropBoundToInputs(const Operation& self, arith::Analyzer* analyzer,
                         const std::unordered_map<const VarNode*, IntSet>& dom_map,
                         std::unordered_map<Tensor, TensorDom>* out_dom_map) const final;
  void GatherBound(const Operation& self, const std::unordered_map<Tensor, TensorDom>& tensor_dom,
                   std::unordered_map<IterVar, Range>* out_dom_map,
                   const Map<FunctionRef, CacheInfo> cacheTensorInfos) const final;
  Stmt BuildRealize(const Stage& stage, const std::unordered_map<IterVar, Range>& realize_map,
                    const Stmt& body) const final;
  Stmt BuildProvide(const Stage& stage, const std::unordered_map<IterVar, Range>& dom_map,
                    const std::unordered_map<std::string, Range>& env_dom_map,
                    const std::unordered_map<std::string, IterVar>& env_var_map,
                    const std::unordered_map<const VarNode*, std::string>& bind_map,
                    const Map<Stage, Array<Stage>>& attach_stages,
                    const Map<Stage, Array<IterVar>>& attach_vars,
                    bool debug_keep_trivial_loop) const final;
  Array<Dimension> GetRootIndexDimensions(size_t val_idx) const;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("tag", &tag);
    v->Visit("attrs", &attrs);
    v->Visit("inputs", &inputs);
    v->Visit("spatial_dimensions_", &spatial_dimensions_);
  }
  static Operation make(std::string name, std::string tag, Map<std::string, ObjectRef> attrs,
                        Array<Array<Tensor>> inputs);

  static constexpr const char* _type_key = "SpecializationEnvelopeOp";
  TVM_DECLARE_FINAL_OBJECT_INFO(SpecializationEnvelopeOpNode, BaseVarDimOpNode);
};

/*!
 * \brief Symbolic scan.
 */
class SingleKernelEnvelopeOpNode : public BaseVarDimOpNode {
 public:
  Array<Tensor> inputs;
  std::vector<const BaseVarDimOpNode*> input_ops;
  Array<Dimension> spatial_dimensions_;
  Array<DimInfo> explicit_dimensions;

  /*! \brief constructor */
  SingleKernelEnvelopeOpNode() {}
  // override behavior.
  int num_outputs() const final;
  Array<IterVar> root_iter_vars() const final;
  DataType output_dtype(size_t i) const final;
  Array<PrimExpr> output_shape(size_t i) const final;
  Dimension GetBaseIndexDimension(size_t val_idx, size_t dim_idx) const final;
  Array<Tensor> InputTensors() const final;
  Array<Tensor> InputTensorsWithUnemitted() const override;
  Operation ReplaceInputs(const Operation& self,
                          const std::unordered_map<Tensor, Tensor>& rmap) const final;
  void PropBoundToInputs(const Operation& self, arith::Analyzer* analyzer,
                         const std::unordered_map<const VarNode*, IntSet>& dom_map,
                         std::unordered_map<Tensor, TensorDom>* out_dom_map) const final;
  void GatherBound(const Operation& self, const std::unordered_map<Tensor, TensorDom>& tensor_dom,
                   std::unordered_map<IterVar, Range>* out_dom_map,
                   const Map<FunctionRef, CacheInfo> cacheTensorInfos) const final;
  Stmt BuildRealize(const Stage& stage, const std::unordered_map<IterVar, Range>& realize_map,
                    const Stmt& body) const final;
  Stmt BuildProvide(const Stage& stage, const std::unordered_map<IterVar, Range>& dom_map,
                    const std::unordered_map<std::string, Range>& env_dom_map,
                    const std::unordered_map<std::string, IterVar>& env_var_map,
                    const std::unordered_map<const VarNode*, std::string>& bind_map,
                    const Map<Stage, Array<Stage>>& attach_stages,
                    const Map<Stage, Array<IterVar>>& attach_vars,
                    bool debug_keep_trivial_loop) const final;
  Array<Dimension> GetRootIndexDimensions(size_t val_idx) const;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("tag", &tag);
    v->Visit("attrs", &attrs);
    v->Visit("inputs", &inputs);
    v->Visit("spatial_dimensions_", &spatial_dimensions_);
  }
  static Operation make(std::string name, std::string tag, Map<std::string, ObjectRef> attrs,
                        Array<Dimension> explicit_dims, Array<Tensor> inputs);

  static constexpr const char* _type_key = "SingleKernelEnvelopeOp";
  // TVM_DECLARE_FINAL_OBJECT_INFO(SingleKernelEnvelopeOpNode, BaseVarDimOpNode);
  TVM_DECLARE_FINAL_OBJECT_INFO(SingleKernelEnvelopeOpNode, OperationNode);
};

/*!
 * \brief External computation that cannot be splitted.
 */
class ExternOpNode : public OperationNode {
 public:
  /*! \brief The input tensors */
  Array<Tensor> inputs;
  /*! \brief Symbolic placeholder representation of inputs */
  Array<Buffer> input_placeholders;
  /*! \brief Symbolic placeholder representation of outputs */
  Array<Buffer> output_placeholders;
  /*! \brief the statement that generates the computation. */
  Stmt body;

  /*! \brief constructor */
  ExternOpNode() {}
  // override functions
  int num_outputs() const final;
  Array<IterVar> root_iter_vars() const final;
  DataType output_dtype(size_t i) const final;
  Array<PrimExpr> output_shape(size_t i) const final;
  Array<Tensor> InputTensors() const final;
  Operation ReplaceInputs(const Operation& self,
                          const std::unordered_map<Tensor, Tensor>& rmap) const final;
  void PropBoundToInputs(const Operation& self, arith::Analyzer* analyzer,
                         const std::unordered_map<const VarNode*, IntSet>& dom_map,
                         std::unordered_map<Tensor, TensorDom>* out_dom_map) const final;
  void GatherBound(const Operation& self, const std::unordered_map<Tensor, TensorDom>& tensor_dom,
                   std::unordered_map<IterVar, Range>* out_dom_map,
                   const Map<FunctionRef, CacheInfo> cacheTensorInfos) const final;
  Stmt BuildRealize(const Stage& stage, const std::unordered_map<IterVar, Range>& realize_map,
                    const Stmt& body) const final;
  Stmt BuildProvide(const Stage& stage, const std::unordered_map<IterVar, Range>& dom_map,
                    const std::unordered_map<std::string, Range>& env_dom_map,
                    const std::unordered_map<std::string, IterVar>& env_var_map,
                    const std::unordered_map<const VarNode*, std::string>& bind_map,
                    const Map<Stage, Array<Stage>>& attach_stages,
                    const Map<Stage, Array<IterVar>>& attach_vars,
                    bool debug_keep_trivial_loop) const final;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("tag", &tag);
    v->Visit("attrs", &attrs);
    v->Visit("inputs", &inputs);
    v->Visit("input_placeholders", &input_placeholders);
    v->Visit("output_placeholders", &output_placeholders);
    v->Visit("body", &body);
  }
  TVM_DLL static Operation make(std::string name, std::string tag,
                                Map<std::string, ObjectRef> attrs, Array<Tensor> inputs,
                                Array<Buffer> input_placeholders, Array<Buffer> output_placeholders,
                                Stmt body);

  static constexpr const char* _type_key = "ExternOp";
  TVM_DECLARE_FINAL_OBJECT_INFO(ExternOpNode, OperationNode);
};

/*!
 * \brief A computation operator that generated by hybrid script.
 */
class HybridOpNode : public OperationNode {
 public:
  /*! \brief The input tensors */
  Array<Tensor> inputs;
  /*! \brief Symbolic placeholder representation of outputs */
  Array<Tensor> outputs;
  /*! \brief The axis of iterations */
  Array<IterVar> axis;
  /*! \brief the statement that generates the computation. This is
   * slightly different from the body in ExternOpNode. All the output
   * tensors keep its own name specified by users in the script.
   * However, when compilation, these tensors will be placed by those
   * actual output tensors. */
  Stmt body;

  /*! \brief constructor */
  HybridOpNode() {}
  // override functions
  int num_outputs() const final;
  Array<IterVar> root_iter_vars() const final;
  DataType output_dtype(size_t i) const final;
  Array<PrimExpr> output_shape(size_t i) const final;
  Array<Tensor> InputTensors() const final;
  Operation ReplaceInputs(const Operation& self,
                          const std::unordered_map<Tensor, Tensor>& rmap) const final;
  void PropBoundToInputs(const Operation& self, arith::Analyzer* analyzer,
                         const std::unordered_map<const VarNode*, IntSet>& dom_map,
                         std::unordered_map<Tensor, TensorDom>* out_dom_map) const final;
  void GatherBound(const Operation& self, const std::unordered_map<Tensor, TensorDom>& tensor_dom,
                   std::unordered_map<IterVar, Range>* out_dom_map,
                   const Map<FunctionRef, CacheInfo> cacheTensorInfos) const final;
  Stmt BuildRealize(const Stage& stage, const std::unordered_map<IterVar, Range>& realize_map,
                    const Stmt& body) const final;
  Stmt BuildProvide(const Stage& stage, const std::unordered_map<IterVar, Range>& dom_map,
                    const std::unordered_map<std::string, Range>& env_dom_map,
                    const std::unordered_map<std::string, IterVar>& env_var_map,
                    const std::unordered_map<const VarNode*, std::string>& bind_map,
                    const Map<Stage, Array<Stage>>& attach_stages,
                    const Map<Stage, Array<IterVar>>& attach_vars,
                    bool debug_keep_trivial_loop) const final;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("tag", &tag);
    v->Visit("attrs", &attrs);
    v->Visit("inputs", &inputs);
    v->Visit("outputs", &outputs);
    v->Visit("axis", &axis);
    v->Visit("body", &body);
  }
  TVM_DLL static Operation make(std::string name, std::string tag,
                                Map<std::string, ObjectRef> attrs, Array<Tensor> inputs,
                                Array<Tensor> outputs, Stmt body);

  static constexpr const char* _type_key = "HybridOp";
  TVM_DECLARE_FINAL_OBJECT_INFO(HybridOpNode, OperationNode);
};

/*!
 * \brief Construct a new Var expression
 * \param name_hint The name hint for the expression
 * \param t The type of the expression
 */
TVM_DLL Var var(std::string name_hint, DataType t = DataType::Int(32));

/*!
 * \brief Create a new IterVar that represents an axis in thread.
 *
 * \param dom Optional, domain of the thread axis.
 * \param tag The thread tag of the axis.
 */
TVM_DLL IterVar thread_axis(Range dom, std::string tag);

/*!
 * \brief Create a new IterVar for reduction operations.
 *
 * \param dom The domain of the reduction axis.
 * \param name The name of the reduction axis.
 */
TVM_DLL IterVar reduce_axis(Range dom, std::string name = "rv");

/*! \brief The compute function to specify the input source of a Tensor */
using FCompute = std::function<PrimExpr(const Array<Var>& i)>;

/*! \brief The compute function to specify the inputs source of Tensors */
using FBatchCompute = std::function<Array<PrimExpr>(const Array<Var>& i)>;

/*! \brief The compute function to specify the input source of a Tensor */
using FComputeMap = std::function<PrimExpr(const Map<Dimension, Var>& i)>;

/*! \brief The compute function to specify the inputs source of Tensors */
using FBatchComputeMap = std::function<Array<PrimExpr>(const Map<Dimension, Var>& i)>;

/*!
 * \brief create a place holder tensor.
 * \param shape The shape of the tensor.
 * \param dtype the data type of the tensor.
 * \param name The name of the Tensor.
 */
TVM_DLL Tensor placeholder(Array<PrimExpr> shape, DataType dtype = DataType::Float(32),
                           std::string name = "placeholder");

/*!
 * \brief Construct a new tensor by computing over shape,
 *  using the computation rule: result_tensor[axis] = fcompute(axis)
 * \param shape Shape of the tensor.
 * \param fcompute The compute function to create the tensor.
 * \param name The optional name of the tensor.
 * \param tag The optional tag of the tensor.
 * \param attrs Optional additional attributes of the compute.
 */
TVM_DLL Tensor compute(Array<PrimExpr> shape, FCompute fcompute, std::string name = "tensor",
                       std::string tag = "", Map<std::string, ObjectRef> attrs = {});

TVM_DLL Array<Tensor> compute(Array<PrimExpr> shape, FBatchCompute fcompute,
                              std::string name = "tensor", std::string tag = "",
                              Map<std::string, ObjectRef> attrs = {});

TVM_DLL Array<Tensor> compute(Array<PrimExpr> shape, FBatchCompute fcompute, FBatchCompute fpred,
                              std::string name, std::string tag, Map<std::string, ObjectRef> attrs,
                              Array<UninterpFun> axis_min_ufs, Array<UninterpFun> axis_extent_ufs,
                              Array<Dimension> dimensions
                              // , Array<Modes> storage_layouts, Modes loop_layout
);
TVM_DLL Array<Tensor> compute(Array<PrimExpr> shape, FBatchComputeMap fcompute,
                              FBatchComputeMap fpred, std::string name, std::string tag,
                              Map<std::string, ObjectRef> attrs, Array<UninterpFun> axis_min_ufs,
                              Array<UninterpFun> axis_extent_ufs, Array<Dimension> dimensions
                              // , Array<Modes> storage_layouts, Modes loop_layout
);

TVM_DLL Array<Tensor> compute(Array<PrimExpr> shape, FBatchComputeMap fcompute,
                              FBatchComputeMap fpred, std::string name, std::string tag,
                              Map<std::string, ObjectRef> attrs, Array<IterVar> axis,
                              Array<Dimension> dimensions
                              // , Array<Modes> storage_layouts, Modes loop_layout
);

/*!
 * \brief Construct new tensors by scan.
 *
 * \param init The intialize tensor of first K steps.
 * \param update The update tensor indicated the updated result after each timestamp.
 * \param state_placeholder The placeholder for the states.
 * \param inputs The inputs to the scan body, this is optional,
 *    but recommended to provide concrete information about scan body.
 * \param name The optional name of the tensor.
 * \param tag The optional tag of the tensor.
 * \param attrs Optional additional attributes of the compute.
 */
TVM_DLL Array<Tensor> scan(Dimension scan_dim, Array<Tensor> init, Array<Tensor> update,
                           Array<Tensor> state_placeholder, Array<Dimension> explicit_loops,
                           Array<Tensor> inputs = Array<Tensor>(), std::string name = "scan",
                           std::string tag = "", Map<std::string, ObjectRef> attrs = {});

// same as compute, specialized for different fcompute function
inline Tensor compute(Array<PrimExpr> shape, std::function<PrimExpr(Var)> f,
                      std::string name = "tensor", std::string tag = "",
                      Map<std::string, ObjectRef> attrs = {}) {
  FCompute fc = [f](const Array<Var>& i) { return f(i[0]); };
  return compute(shape, fc, name, tag, attrs);
}
inline Tensor compute(Array<PrimExpr> shape, std::function<PrimExpr(Var, Var)> f,
                      std::string name = "tensor", std::string tag = "",
                      Map<std::string, ObjectRef> attrs = {}) {
  FCompute fc = [f](const Array<Var>& i) { return f(i[0], i[1]); };
  return compute(shape, fc, name, tag, attrs);
}
inline Tensor compute(Array<PrimExpr> shape, std::function<PrimExpr(Var, Var, Var)> f,
                      std::string name = "tensor", std::string tag = "",
                      Map<std::string, ObjectRef> attrs = {}) {
  FCompute fc = [f](const Array<Var>& i) { return f(i[0], i[1], i[2]); };
  return compute(shape, fc, name, tag, attrs);
}
inline Tensor compute(Array<PrimExpr> shape, std::function<PrimExpr(Var, Var, Var, Var)> f,
                      std::string name = "tensor", std::string tag = "",
                      Map<std::string, ObjectRef> attrs = {}) {
  FCompute fc = [f](const Array<Var>& i) { return f(i[0], i[1], i[2], i[3]); };
  return compute(shape, fc, name, tag, attrs);
}

// inline function.
inline const OperationNode* Operation::operator->() const {
  return static_cast<const OperationNode*>(get());
}
}  // namespace te
}  // namespace tvm
#endif  // TVM_TE_OPERATION_H_
