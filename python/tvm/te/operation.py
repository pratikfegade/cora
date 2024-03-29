# Licensed to the Apache Software Foundation (ASF) under one
# or more ibutor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
""" Operation class for computation declaration."""
# pylint: disable=invalid-name
from numbers import Integral as _Integral

import sys
import tvm._ffi
import tvm.tir
import tvm.tir._ffi_api

from tvm._ffi.base import string_types
from tvm.runtime import convert
# from tvm.arith import UninterpFun

from . import tag as _tag
from . import tensor as _tensor
from . import _ffi_api
from tvm.tir import Modes
from tvm.tir import LFunsWrapper

class Dimension(tvm.runtime.Object):
    pass

@tvm._ffi.register_object("te.Dimension")
class RangeDimension(Dimension):
    """Represent set of continuous interval [min_value, max_value]

    Parameters
    ----------
    min_value : PrimExpr
        The minimum value in the interval.

    max_value : PrimExpr
        The maximum value in the interval.
    """
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.__init_handle_by_constructor__(_ffi_api.RangeDimension, name)

    def __str__(self):
        return 'Dimension('+self.name+')'


@tvm._ffi.register_object("te.Dimension")
class ScanDimension(Dimension):
    """Represent set of continuous interval [min_value, max_value]

    Parameters
    ----------
    min_value : PrimExpr
        The minimum value in the interval.

    max_value : PrimExpr
        The maximum value in the interval.
    """
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.__init_handle_by_constructor__(_ffi_api.ScanDimension, name)

    def __str__(self):
        return 'Dimension('+self.name+')'

def placeholder(shape, dtype=None, name="placeholder"):
    """Construct an empty tensor object.

    Parameters
    ----------
    shape: Tuple of Expr
        The shape of the tensor

    dtype: str, optional
        The data type of the tensor

    name: str, optional
        The name hint of the tensor

    Returns
    -------
    tensor: Tensor
        The created tensor
    """
    shape = (shape,) if isinstance(shape, tvm.tir.PrimExpr) else shape
    dtype = "float32" if dtype is None else dtype
    return _ffi_api.Placeholder(
        shape, dtype, name)


def create_or_return_uf(expr):
    if isinstance(expr, tvm.tir.UninterpFun):
        return expr
    else:
        ret = tvm.tir.UninterpFun("uf", (expr, expr), [], lambda: expr)
        return ret


def ragged_placeholder(dense_shape, dimensions, loop_extent_ufs, dtype=None,
                       name="placeholder", width_ufs=None, aggregate_ufs={}):
    layout = None
    if width_ufs is not None:
        layout = Modes.storage_layout(dimensions, dense_shape, width_ufs, aggregate_ufs)

    if isinstance(loop_extent_ufs, LFunsWrapper): loop_extent_ufs = loop_extent_ufs.get_ufs()
    ret = indirect_placeholder_integrated(dense_shape, dimensions, list(zip(dimensions, loop_extent_ufs)),
                                          dtype, name, layout)
    return ret

def indirect_placeholder(shape, self_dims, loop_extent_dims, idx_expr_dims, dtype=None, name="placeholder", layout=None):
    return indirect_placeholder_integrated(shape, self_dims, loop_extent_dims + idx_expr_dims, dtype, name, layout)

def indirect_placeholder_integrated(shape, self_dims, dim_ufs, dtype=None, name="placeholder", layout=None):
    all_vars = []
    all_dims = []
    all_ufs = []
    for dim_uf in dim_ufs:
        dim = dim_uf[0]
        all_ufs.append(None)
        if len(dim_uf) == 2:
            _, max_val_uf_orig = dim_uf
            max_val_uf = create_or_return_uf(max_val_uf_orig)

            max_val = tvm.tir.Call("int32", max_val_uf.fname, [v.var for v in all_vars],
                                  2, max_val_uf, 0, arg_dims = all_dims)
            iter_var = tvm.tir.IterVar((0, max_val), 'i' + name + str(len(all_vars)), 0)
            all_vars.append(iter_var)
            all_dims.append(dim)
        else:
            _, min_uf_orig, max_val_uf_orig = dim_uf
            min_uf = create_or_return_uf(min_uf_orig)
            max_val_uf = create_or_return_uf(max_val_uf_orig)

            dom_min = tvm.tir.Call("int32", min_uf.fname, [v.var for v in all_vars],
                                   2, min_uf, 0, arg_dims = all_dims)

            dom_max_val = tvm.tir.Call("int32", max_val_uf.fname, [v.var for v in all_vars],
                                      2, max_val_uf, 0, arg_dims = all_dims)
            iter_var = tvm.tir.IterVar((dom_min, dom_max_val), 'i' + name + str(len(all_vars)), 0)
            all_vars.append(iter_var)
            all_dims.append(dim)

    shape = (shape,) if isinstance(shape, tvm.tir.PrimExpr) else shape
    dtype = "float32" if dtype is None else dtype
    return _ffi_api.IndirectPlaceholder(
        shape, layout, self_dims, all_dims, all_vars, all_ufs, dtype, name)

def compute(shape, fcompute, name="compute", tag="", attrs=None):
    """Construct a new tensor by computing over the shape domain.

    The compute rule is result[axis] = fcompute(axis)

    Parameters
    ----------
    shape: Tuple of Expr
        The shape of the tensor

    fcompute: lambda function of indices-> value
        Specifies the input source expression

    name: str, optional
        The name hint of the tensor

    tag: str, optional
        Additional tag information about the compute.

    attrs: dict, optional
        The additional auxiliary attributes about the compute.

    Returns
    -------
    tensor: Tensor
        The created tensor
    """
    if _tag.TagScope.get_current() is not None:
        if tag != "":
            raise ValueError("nested tag is not allowed for now")
        tag = _tag.TagScope.get_current().tag
    shape = (shape,) if isinstance(shape, tvm.tir.PrimExpr) else shape
    # for python3
    shape = tuple([int(s) if isinstance(s, float) else s for s in shape])
    ndim = len(shape)
    code = fcompute.__code__

    out_ndim = ndim
    if code.co_argcount == 0:
        arg_names = ["i%d" % i for i in range(ndim)]
    else:
        arg_names = code.co_varnames[:code.co_argcount]
        out_ndim = code.co_argcount

    if out_ndim != len(arg_names):
        raise ValueError("fcompute do not match dimension, ndim=%d" % ndim)

    dim_var = [tvm.tir.IterVar((0, s), x, 0) for x, s in zip(arg_names, shape[:out_ndim])]
    body = fcompute(*[v.var for v in dim_var])

    if isinstance(body, _tensor.TensorIntrinCall):
        for i, s in enumerate(shape[out_ndim:]):
            var_name = "ax" + str(i)
            dim_var.append(tvm.tir.IterVar((0, s), var_name, 4))
        op_node = _ffi_api.TensorComputeOp(name,
                                           tag,
                                           dim_var,
                                           body.reduce_axis,
                                           out_ndim,
                                           body.intrin,
                                           body.tensors,
                                           body.regions,
                                           body.scalar_inputs)
    else:
        if not isinstance(body, (list, tuple)):
            body = [body]
        body = convert(body)
        op_node = _ffi_api.ComputeOp(
            name, tag, attrs, dim_var, body)

    num = op_node.num_outputs
    outputs = tuple(op_node.output(i) for i in range(num))
    return outputs[0] if num == 1 else outputs


def ragged_compute(dense_shape, dimensions, loop_extent_ufs, fcompute, reduce_axis_ufs=None, fpred=None, name="compute",
                   tag="", attrs=None, loop_aggregate_ufs=None, width_uf_lists=None, aggregate_uf_lists=None, num_outputs=1):
    storage_layouts = None
    if width_uf_lists is not None:
        if width_uf_lists is None: width_uf_lists = [[]] * num_outputs
        if aggregate_uf_lists is None: aggregate_uf_lists = [{}] * num_outputs
        # storage_layouts = [Modes(dimensions, dense_shape, width_ufs, aggregate_ufs) for width_ufs,
        storage_layouts = [Modes.storage_layout(dimensions, dense_shape, width_ufs, aggregate_ufs) for width_ufs,
                   aggregate_ufs in zip(width_uf_lists, aggregate_uf_lists)]

    mode_loop_extent_ufs = []
    mode_loop_min_ufs = []
    if isinstance(loop_extent_ufs, LFunsWrapper): loop_extent_ufs = loop_extent_ufs.get_ufs()
    for uf in loop_extent_ufs:
        if isinstance(uf, tvm.tir.UninterpFun):
            mode_loop_min_ufs.append(tvm.tir.UninterpFun.from_constant('zero', 0, 'l'))
            mode_loop_extent_ufs.append(uf)
        else:
            mode_loop_min_ufs.append(uf[0])
            mode_loop_extent_ufs.append(uf[1])

    # loop_layout = Modes(dimensions, dense_shape, mode_loop_extent_ufs, loop_aggregate_ufs, loop_layout = True)
    loop_layout = Modes.loop_layout(dimensions, dense_shape, mode_loop_min_ufs, mode_loop_extent_ufs)

    output_shape = dense_shape
    dim_ufs = list()

    if _tag.TagScope.get_current() is not None:
        if tag != "":
            raise ValueError("nested tag is not allowed for now")
        tag = _tag.TagScope.get_current().tag
    output_shape = (output_shape,) if isinstance(output_shape, tvm.tir.PrimExpr) else output_shape
    # for python3
    output_shape = tuple([int(s) if isinstance(s, float) else s for s in output_shape])

    code = fcompute.__code__

    out_ndim = len(output_shape)
    if code.co_argcount > 1 and reduce_axis_ufs is None:
        raise ValueError("Ill-formed body lambda with more than one argument")

    if out_ndim != len(dimensions):
        raise ValueError("Dimensions of the output do not match the number of self dimensions given")

    all_dims = []
    axis = []
    dim_var_map = {}
    for dim, ufs in zip(dimensions, loop_extent_ufs):
        min_uf, max_uf = None, None
        if isinstance(ufs, (list, tuple)):
            min_uf, max_uf = ufs
        else:
            max_uf = ufs

        dom_max = tvm.tir.Call("int32", max_uf.fname, [v.var for v in axis],
                               2, max_uf, 0, arg_dims = all_dims)
        if min_uf:
            dom_min = tvm.tir.Call("int32", min_uf.fname, [v.var for v in axis],
                                   2, min_uf, 0, arg_dims = all_dims)
        else:
            dom_min = 0

        iter_var = tvm.tir.IterVar((dom_min, dom_max), 'i' + name + str(len(axis)), 0)

        axis.append(iter_var)
        all_dims.append(dim)
        dim_var_map[dim] = iter_var


    if reduce_axis_ufs is not None:
        reduce_ivs = {}
        for iv_name, uf in reduce_axis_ufs:
            dom_max = tvm.tir.Call("int32", uf.fname, [v.var for v in axis],
                                   2, uf, 0, arg_dims = all_dims)
            iter_var = reduce_axis((0, dom_max), iv_name)
            dim_var_map[iv_name] = iter_var
            reduce_ivs[iv_name] = iter_var
        body = fcompute({k: v.var for k, v in dim_var_map.items()}, reduce_ivs)
    else:
        body = fcompute({k: v.var for k, v in dim_var_map.items()})

    pred = fpred({k: v.var for k, v in dim_var_map.items()}) if fpred is not None else [tvm.tir.IntImm('uint1', 1)]

    if isinstance(body, _tensor.TensorIntrinCall):
        for i, s in enumerate(loop_domains[out_ndim:]):
            var_name = "ax" + str(i)
            loop_var.append(tvm.tir.IterVar((0, s), var_name, 4))
        op_node = _ffi_api.TensorComputeOp(name, tag, loop_var, body.reduce_axis, out_ndim,
                                           body.intrin, body.tensors, body.regions, body.scalar_inputs)
    else:
        if not isinstance(body, (list, tuple)):
            body = [body]
        body = convert(body)
        if not isinstance(pred, (list, tuple)):
            pred = [pred]
        pred = convert(pred)
        op_node = _ffi_api.ComputeOp(name, tag, attrs, axis, dimensions, output_shape,
                                     storage_layouts, loop_layout,
                                     body, pred)

    num = op_node.num_outputs
    outputs = tuple(op_node.output(i) for i in range(num))
    return outputs[0] if num == 1 else outputs


def indirect_compute(output_shape, self_dims, loop_domains, idx_expr_ufs, fcompute,
                     reduce_axis_ufs=None, fpred = None, name="compute", tag="", attrs=None):
    return indirect_compute_integrated(output_shape, self_dims, loop_domains + idx_expr_ufs,
                                       fcompute, reduce_axis_ufs, fpred, name, tag, attrs)

def indirect_compute_integrated(output_shape, dimensions, dim_ufs, fcompute, reduce_axis_ufs=None, fpred = None,
                                name="compute", tag="", attrs=None, storage_layouts=None, loop_layout=None):
    if _tag.TagScope.get_current() is not None:
        if tag != "":
            raise ValueError("nested tag is not allowed for now")
        tag = _tag.TagScope.get_current().tag
    output_shape = (output_shape,) if isinstance(output_shape, tvm.tir.PrimExpr) else output_shape
    # for python3
    output_shape = tuple([int(s) if isinstance(s, float) else s for s in output_shape])

    code = fcompute.__code__

    out_ndim = len(output_shape)
    if code.co_argcount > 1 and reduce_axis_ufs is None:
        raise ValueError("Ill-formed body lambda with more than one argument")

    if out_ndim != len(self_dims):
        raise ValueError("Dimensions of the output do not match the number of self dimensions given")

    all_vars = []
    all_dims = []
    axis = []
    dim_var_map = {}
    for dim_uf in dim_ufs:
        dim = dim_uf[0]
        if len(dim_uf) == 2:
            _, max_uf_orig = dim_uf
            max_uf = create_or_return_uf(max_uf_orig)

            dom_max = tvm.tir.Call("int32", max_uf.fname, [v.var for v in all_vars],
                                  2, max_uf, 0, arg_dims = all_dims)
            iter_var = tvm.tir.IterVar((0, dom_max), 'i' + name + str(len(all_vars)), 0)
        else:
            _, min_uf_orig, max_uf_orig = dim_uf
            min_uf = create_or_return_uf(min_uf_orig)
            max_uf = create_or_return_uf(max_uf_orig)

            dom_min = tvm.tir.Call("int32", min_uf.fname, [v.var for v in all_vars],
                                   2, min_uf, 0, arg_dims = all_dims)

            dom_max = tvm.tir.Call("int32", max_uf.fname, [v.var for v in all_vars],
                                      2, max_uf, 0, arg_dims = all_dims)
            iter_var = tvm.tir.IterVar(tvm.ir.Range(dom_min, dom_max),
                                       'i' + name + str(len(all_vars)), 0)
        all_vars.append(iter_var)
        axis.append(iter_var)
        all_dims.append(dim)
        dim_var_map[dim] = iter_var


    if reduce_axis_ufs is not None:
        reduce_ivs = {}
        for iv_name, uf in reduce_axis_ufs:
            dom_max = tvm.tir.Call("int32", uf.fname, [v.var for v in all_vars],
                                   2, uf, 0, arg_dims = all_dims)
            iter_var = reduce_axis((0, dom_max), iv_name)
            dim_var_map[iv_name] = iter_var
            reduce_ivs[iv_name] = iter_var
        body = fcompute({k: v.var for k, v in dim_var_map.items()}, reduce_ivs)
    else:
        body = fcompute({k: v.var for k, v in dim_var_map.items()})

    pred = fpred({k: v.var for k, v in dim_var_map.items()}) if fpred is not None else [tvm.tir.IntImm('uint1', 1)]

    if isinstance(body, _tensor.TensorIntrinCall):
        for i, s in enumerate(loop_domains[out_ndim:]):
            var_name = "ax" + str(i)
            loop_var.append(tvm.tir.IterVar((0, s), var_name, 4))
        op_node = _ffi_api.TensorComputeOp(name,
                                           tag,
                                           loop_var,
                                           body.reduce_axis,
                                           out_ndim,
                                           body.intrin,
                                           body.tensors,
                                           body.regions,
                                           body.scalar_inputs)
    else:
        if not isinstance(body, (list, tuple)):
            body = [body]
        body = convert(body)
        if not isinstance(pred, (list, tuple)):
            pred = [pred]
        pred = convert(pred)
        op_node = _ffi_api.ComputeOp(name, tag, attrs, axis, dimensions, output_shape,
                                     storage_layouts, loop_layout, all_vars,
                                     body, pred)

    num = op_node.num_outputs
    outputs = tuple(op_node.output(i) for i in range(num))
    return outputs[0] if num == 1 else outputs


def scan(init, update, state_placeholder, inputs=None, name="scan", tag="", attrs=None):
    """Construct new tensors by scanning over axis.

    Parameters
    ----------
    init: Tensor or list of Tensor
        The initial condition of first init.shape[0] timestamps

    update: Tensor or list of Tensor
        The update rule of the scan given by symbolic tensor.

    state_placeholder: Tensor or list of Tensor
        The placeholder variables used by update.

    inputs: Tensor or list of Tensor, optional
        The list of inputs to the scan. This is not required, but can
        be useful for the compiler to detect scan body faster.

    name: str, optional
        The name hint of the tensor

    tag: str, optional
        Additonal tag information about the compute.

    attrs: dict, optional
        The additional auxiliary attributes about the compute.

    Returns
    -------
    tensor: Tensor or list of Tensors
        The created tensor or tuple of tensors it it contains multiple outputs.

    Example
    -------
    .. code-block:: python

      # The following code is equivalent to numpy.cumsum
      m = tvm.var("m")
      n = tvm.var("n")
      X = tvm.placeholder((m, n), name="X")
      s_state = tvm.placeholder((m, n))
      s_init = tvm.compute((1, n), lambda _, i: X[0, i])
      s_update = tvm.compute((m, n), lambda t, i: s_state[t-1, i] + X[t, i])
      res = tvm.scan(s_init, s_update, s_state, X)
    """
    if _tag.TagScope.get_current() is not None:
        if tag != "":
            raise ValueError("nested tag is not allowed for now")
        tag = _tag.TagScope.get_current().tag
    if isinstance(init, _tensor.Tensor):
        init = [init]
    if isinstance(update, _tensor.Tensor):
        update = [update]
    if isinstance(state_placeholder, _tensor.Tensor):
        state_placeholder = [state_placeholder]
    if isinstance(inputs, _tensor.Tensor):
        inputs = [inputs]
    if inputs is None:
        inputs = []
    if len(init) != len(update) or len(init) != len(state_placeholder):
        raise ValueError("init, update, state_placeholder must have same length")
    axis = tvm.tir.IterVar((init[0].shape[0], update[0].shape[0]), "%s.idx" % name, 3)
    op = _ffi_api.ScanOp(name, tag, attrs,
                         axis, init, update,
                         state_placeholder, inputs)
    res = [op.output(i) for i in range(len(update))]
    return res[0] if len(res) == 1 else res

def indirect_scan(range_min_uf, range_max_uf, scan_dim, init, update, state_placeholder,
                  explicit_dim_ufs = [], init_separate = False, inputs=None, name="scan", tag="", attrs=None):
    if _tag.TagScope.get_current() is not None:
        if tag != "":
            raise ValueError("nested tag is not allowed for now")
        tag = _tag.TagScope.get_current().tag
    if isinstance(init, _tensor.Tensor):
        init = [init]
    if isinstance(update, _tensor.Tensor):
        update = [update]
    if isinstance(state_placeholder, _tensor.Tensor):
        state_placeholder = [state_placeholder]
    if isinstance(inputs, _tensor.Tensor):
        inputs = [inputs]
    if inputs is None:
        inputs = []
    if len(init) != len(update) or len(init) != len(state_placeholder):
        raise ValueError("init, update, state_placeholder must have same length")

    exp_min_ufs = []
    exp_dims = []
    exp_max_ufs = []
    for dim_uf in explicit_dim_ufs:
        exp_dims.append(dim_uf[0])
        if len(dim_uf) == 2:
            exp_min_ufs.append(tvm.tir.UninterpFun.from_constant('z', 0))
            exp_max_ufs.append(dim_uf[1])
        else:
            exp_min_ufs.append(dim_uf[1])
            exp_max_ufs.append(dim_uf[2])

    op = _ffi_api.ScanOp(name, tag, attrs, range_min_uf,
                         range_max_uf, scan_dim, init_separate, init, update,
                         state_placeholder, inputs, exp_dims,
                         exp_min_ufs, exp_max_ufs)
    res = [op.output(i) for i in range(len(update))]
    return res[0] if len(res) == 1 else res

def conditional(condition_uf, from_then, then_case, from_else,
                else_case, explicit_dim_ufs = [], name="scan", tag="", attrs=None):
    if _tag.TagScope.get_current() is not None:
        if tag != "":
            raise ValueError("nested tag is not allowed for now")
        tag = _tag.TagScope.get_current().tag
    if isinstance(then_case, _tensor.Tensor):
        then_case = [then_case]
    if isinstance(else_case, _tensor.Tensor):
        else_case = [else_case]
    if len(then_case) != len(else_case):
        raise ValueError("then and else cases must have same length")

    exp_min_ufs = []
    exp_dims = []
    exp_max_ufs = []
    for dim_uf in explicit_dim_ufs:
        exp_dims.append(dim_uf[0])
        if len(dim_uf) == 2:
            exp_min_ufs.append(tvm.tir.UninterpFun.from_constant('z', 0))
            exp_max_ufs.append(dim_uf[1])
        else:
            exp_min_ufs.append(dim_uf[1])
            exp_max_ufs.append(dim_uf[2])

    op = _ffi_api.ConditionalOp(name, tag, attrs,
                                condition_uf, from_then, then_case, from_else,
                                else_case, exp_dims, exp_min_ufs, exp_max_ufs)
    res = [op.output(i) for i in range(len(then_case))]
    return res[0] if len(res) == 1 else res

def specialization_envelope(scans, inputs=None, name="scan", tag="", attrs=None):
    """Construct new tensors by scanning over axis.

    Parameters
    ----------
    name: str, optional
        The name hint of the tensor

    tag: str, optional
        Additonal tag information about the compute.

    attrs: dict, optional
        The additional auxiliary attributes about the compute.

    Returns
    -------
    tensor: Tensor or list of Tensors
        The created tensor or tuple of tensors it it contains multiple outputs.
    """
    op = _ffi_api.SpecializationEnvelopeOp(name, tag, attrs,
                                 scans)
    res = [op.output(i) for i in range(len(scans[0]))]
    return res[0] if len(res) == 1 else res


def extern(shape,
           inputs,
           fcompute,
           name="extern",
           dtype=None,
           in_buffers=None,
           out_buffers=None,
           tag="",
           attrs=None):
    """Compute several tensor via extern function.

    Parameters
    ----------
    shape: tuple or list of tuples.
        The shape of the outputs.

    inputs: list of Tensor
        The inputs

    fcompute: lambda function of inputs, outputs-> stmt
        Specifies the IR statement to do the computation.
        See the following note for function signature of fcompute

        .. note::
             **Parameters**

             - **ins** (list of :any:`Buffer`) - Placeholder for each inputs
             - **outs** (list of :any:`Buffer`) - Placeholder for each outputs

             **Returns**

             - **stmt** (:any:`Stmt`) - The statement that carries out array computation.

    name: str, optional
        The name hint of the tensor

    dtype: str or list of str, optional
        The data types of outputs,
        by default dtype will be same as inputs.

    in_buffers: Buffer or list of Buffer, optional
        Input buffers.

    out_buffers: Buffer or list of Buffers, optional
        Output buffers.


    tag: str, optional
        Additonal tag information about the compute.

    attrs: dict, optional
        The additional auxiliary attributes about the compute.

    Returns
    -------
    tensor: Tensor or list of Tensors
        The created tensor or tuple of tensors it it contains multiple outputs.

    Example
    -------
    In the code below, C is generated by calling external PackedFunc
    `tvm.contrib.cblas.matmul`

    .. code-block:: python

        A = tvm.placeholder((n, l), name="A")
        B = tvm.placeholder((l, m), name="B")
        C = tvm.extern((n, m), [A, B],
                       lambda ins, outs: tvm.call_packed(
                          "tvm.contrib.cblas.matmul",
                            ins[0], ins[1], outs[0], 0, 0), name="C")
    """
    if _tag.TagScope.get_current() is not None:
        if tag != "":
            raise ValueError("nested tag is not allowed for now")
        tag = _tag.TagScope.get_current().tag
    shape = (shape,) if isinstance(shape, (tvm.tir.PrimExpr, _Integral)) else shape
    if shape == () or isinstance(shape[0], (tvm.tir.PrimExpr, _Integral)):
        shape = [shape]
    if in_buffers is not None:
        in_buffers = [in_buffers] if not isinstance(in_buffers, list) else in_buffers
        if len(inputs) != len(in_buffers):
            raise RuntimeError("Number of inputs and in_buffers mismatch: %d vs %d."
                               % (len(inputs), len(in_buffers)))
    if out_buffers is not None:
        out_buffers = [out_buffers] if not isinstance(out_buffers, list) else out_buffers
        if len(shape) != len(out_buffers):
            raise RuntimeError("Number of outputs and out_buffers mismatch: %d vs %d."
                               % (len(shape), len(out_buffers)))
    input_placeholders = in_buffers or []
    output_placeholders = out_buffers or []
    types = set()
    for t in inputs:
        if not isinstance(t, _tensor.Tensor):
            raise ValueError("expect inputs to be tensor")
        if in_buffers is None:
            input_placeholders.append(
                tvm.tir.decl_buffer(t.shape, t.dtype, t.op.name))
        types.add(t.dtype)

    if dtype is None:
        if len(types) != 1:
            raise ValueError("Cannot infer output type, please provide dtype argument")
        infered_type = types.pop()
        dtype = [infered_type for _ in shape]
    if isinstance(dtype, str):
        dtype = [dtype]

    if out_buffers is None:
        for shp, dt in zip(shape, dtype):
            output_placeholders.append(tvm.tir.decl_buffer(shp, dt, name))
    body = fcompute(input_placeholders, output_placeholders)
    if isinstance(body, tvm.tir.PrimExpr):
        body = tvm.tir.Evaluate(body)

    op = _ffi_api.ExternOp(name, tag, attrs,
                           inputs, input_placeholders,
                           output_placeholders, body)
    res = [op.output(i) for i in range(len(output_placeholders))]
    return res[0] if len(res) == 1 else res


def var(name="tindex", dtype="int32"):
    """Create a new variable with specified name and dtype

    Parameters
    ----------
    name : str
        The name

    dtype : str
        The data type

    Returns
    -------
    var : Var
        The result symbolic variable.
    """
    return tvm.tir.Var(name, dtype)


def size_var(name="size", dtype="int32"):
    """Create a new variable represents a tensor shape size, which is non-negative.

    Parameters
    ----------
    name : str
        The name

    dtype : str
        The data type

    Returns
    -------
    var : SizeVar
        The result symbolic shape variable.
    """
    return tvm.tir.SizeVar(name, dtype)


def thread_axis(dom=None, tag="", name=""):
    """Create a new IterVar to represent thread index.

    Parameters
    ----------
    dom : Range or str
        The domain of iteration
        When str is passed, dom is set to None and str is used as tag

    tag : str, optional
        The thread tag

    name : str, optional
        The name of the var.

    Returns
    -------
    axis : IterVar
        The thread itervar.
    """
    if isinstance(dom, string_types):
        tag, dom = dom, None
    if not tag:
        raise ValueError("tag must be given as Positional or keyword argument")
    name = name if name else tag
    return tvm.tir.IterVar(dom, name, 1, tag)


def reduce_axis(dom, name="rv"):
    """Create a new IterVar for reduction.

    Parameters
    ----------
    dom : Range
        The domain of iteration.

    name : str
        The name of the variable.

    Returns
    -------
    axis : IterVar
        An iteration variable representing the value.
    """
    return tvm.tir.IterVar(dom, name, 2)
