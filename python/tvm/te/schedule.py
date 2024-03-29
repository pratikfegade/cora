# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
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
# pylint: disable=unused-import
"""The computation schedule api of TVM."""
import tvm._ffi
from tvm._ffi.base import string_types

from tvm.runtime import Object, convert
from tvm.ir import container as _container
from tvm.tir import IterVar, Buffer, UninterpFun, Modes

from . import tensor as _tensor
from . import _ffi_api


@tvm._ffi.register_object
class Split(Object):
    """Split operation on axis."""


@tvm._ffi.register_object
class Fuse(Object):
    """Fuse operation on axis."""


@tvm._ffi.register_object
class Singleton(Object):
    """Singleton axis."""

def fuse_ragged_axis(input_tensors, output_tensor, outer_dim, inner_dim, fused_dim, fused_extent):
    # input_tensors = [t.op for t in input_tensors]
    # output_tensor = output_tensor.op
    return _ffi_api.FuseRaggedAxis(input_tensors, output_tensor, outer_dim, inner_dim, fused_dim, fused_extent)

def create_schedule(ops):
    """Create a schedule for list of ops

    Parameters
    ----------
    ops : list of Operations
        The source expression.

    Returns
    -------
    sch : schedule.Schedule
        The created schedule.
    """
    if not isinstance(ops, (list, _container.Array)):
        ops = [ops]
    return _ffi_api.CreateSchedule(ops)


@tvm._ffi.register_object
class Schedule(Object):
    """Schedule for all the stages."""
    def __getitem__(self, k):
        if isinstance(k, _tensor.Tensor):
            k = k.op
        if not isinstance(k, _tensor.Operation):
            raise ValueError("Expect schedule key to be Tensor or Operation " + str(type(k)))
        if k not in self.stage_map:
            raise ValueError("Cannot find the operation %s in schedule" % (str(k)))
        return self.stage_map[k]

    def normalize(self):
        """Build a normalized schedule from the current schedule.

        Insert necessary rebase to make certain iter var to start from 0.
        This is needed before bound inference and followup step.

        Returns
        -------
        sch : Schedule
            The normalized schedule.
        """
        return _ffi_api.ScheduleNormalize(self)

    def create_group(self, outputs, inputs, include_inputs=False):
        """Create stage group by giving output and input boundary.

        The operators between outputs and inputs are placed as member of group.
        outputs are include in the group, while inputs are not included.

        Parameters
        ----------
        outputs : list of Tensors
            The outputs of the group.

        inputs : list of Tensors
            The inputs of the group.

        include_inputs : boolean, optional
            Whether include input operations in the group if they are used by outputs.

        Returns
        -------
        group : Stage
            A virtual stage represents the group, user can use compute_at to move
            the attachment point of the group.
        """
        if isinstance(outputs, _tensor.Tensor):
            outputs = [outputs]
        if isinstance(inputs, _tensor.Tensor):
            inputs = [inputs]
        return _ffi_api.ScheduleCreateGroup(
            self, outputs, inputs, include_inputs)

    def cache_read_opaque(self, tensor, scope, readers, suffix = ''):
        """Create a cache read of original tensor for readers.
        This will mutate the body of the readers.
        A new cache stage will be created for the tensor.
        Call this before doing any split/fuse schedule.
        Parameters
        ----------
        tensor : Tensor
            The tensor to be cached.
        scope : str
            The scope of cached
        readers : list of Tensor or Operation
            The readers to read the cache.
        Returns
        -------
        cache : Tensor
            The created cache tensor.
        """
        if isinstance(readers, (_tensor.Tensor, _tensor.Operation)):
            readers = [readers]
        readers = [t.op if isinstance(t, _tensor.Tensor) else t for t in readers]
        return _ffi_api.ScheduleCacheReadOpaque(self, tensor, scope, readers, suffix)

    def cache_read_opaque_all_readers(self, tensor, scope, suffix = ''):
        """Create a cache read of original tensor for readers.
        This will mutate the body of the readers.
        A new cache stage will be created for the tensor.
        Call this before doing any split/fuse schedule.
        Parameters
        ----------
        tensor : Tensor
            The tensor to be cached.
        scope : str
            The scope of cached
        Returns
        -------
        cache : Tensor
            The created cache tensor.
        """
        return _ffi_api.ScheduleCacheReadOpaqueAllReaders(self, tensor, scope, suffix)

    def cache_read(self, tensor, scope, readers, suffix = '', vanilla = False,
                   layouts=None, loop_layout=None, axis_mirror_loop_layout=False):
        """Create a cache read of original tensor for readers.

        This will mutate the body of the readers.
        A new cache stage will be created for the tensor.
        Call this before doing any split/fuse schedule.

        Parameters
        ----------
        tensor : Tensor
            The tensor to be cached.
        scope : str
            The scope of cached
        readers : list of Tensor or Operation
            The readers to read the cache.

        Returns
        -------
        cache : Tensor
            The created cache tensor.
        """
        layout_ufs = layouts
        layouts = []
        if layout_ufs:
            if isinstance(layout_ufs, list):
                if tensor.op.num_outputs == 1:
                    layouts.append(Modes.storage_layout(tensor.op.get_root_index_dimensions(tensor.value_index),
                                                        [f.frange[1] for f in layout_ufs], layout_ufs, {}))
                else:
                    for uf_list in layout_ufs:
                        layouts.append(Modes.storage_layout(tensor.op.get_root_index_dimensions(tensor.value_index),
                                             [f.frange[1] for f in uf_list], uf_list, {}))
            elif layout_ufs == "dense":
                for i in range(tensor.op.num_outputs):
                    l_funs = [UninterpFun.from_constant('f' + str(i), shp, 'l') for shp in tensor.shape]
                    layouts.append(Modes.storage_layout(tensor.op.get_root_index_dimensions(tensor.value_index),
                                        tensor.shape, l_funs, {}))

        loop_layout_ufs = loop_layout
        loop_layout = None
        if loop_layout_ufs:
            l_maxes = []
            l_mins = []
            l_exts = []
            for uf in loop_layout_ufs:
                if isinstance(uf, tuple):
                    min_uf, max_uf = uf
                else:
                    min_uf = tvm.tir.UninterpFun.from_constant('z', 0, 'l')
                    max_uf = uf
                l_mins.append(min_uf)
                l_exts.append(max_uf)
                l_maxes.append(max_uf.frange[0] + max_uf.frange[1])
            # print(tensor)
            # print('   ', l_mins)
            # print('   ', l_exts)
            # print('   ', l_maxes)
            loop_layout = Modes.loop_layout(tensor.op.get_root_index_dimensions(tensor.value_index), l_maxes, l_mins, l_exts)

        if isinstance(readers, (_tensor.Tensor, _tensor.Operation)):
            readers = [readers]
        readers = [t.op if isinstance(t, _tensor.Tensor) else t for t in readers]
        return _ffi_api.ScheduleCacheRead(self, tensor, scope, readers, suffix, vanilla,
                                          layouts, loop_layout, axis_mirror_loop_layout)

    def single_kernel(self, inputs, outputs, threads, name, tag="", attrs=None, include_inputs=False):
        op = _ffi_api.ScheduleSingleKernel(self, name, tag, attrs, inputs, outputs, include_inputs, threads)
        res = [op.output(i) for i in range(len(outputs))]
        return res[0] if len(res) == 1 else res

    def split_for_bin_packing(self, inputs, output, to_split, include_inputs=True):
        graphs = _ffi_api.SplitForBinPacking(self, inputs, output, to_split, include_inputs)
        return [[o.output(0) for o in ops] for ops in graphs]

    def unify(self, ops, explicit_dims, name, tag="", attrs=None):
        op = _ffi_api.ScheduleUnify(self, name, tag, attrs, ops, explicit_dims)
        res = [op.output(i) for i in range(len(ops))]
        return res[0] if len(res) == 1 else res

    def cache_read_opaque(self, tensor, scope, readers, suffix = ''):
        """Create a cache read of original tensor for readers.

        This will mutate the body of the readers.
        A new cache stage will be created for the tensor.
        Call this before doing any split/fuse schedule.

        Parameters
        ----------
        tensor : Tensor
            The tensor to be cached.
        scope : str
            The scope of cached
        readers : list of Tensor or Operation
            The readers to read the cache.

        Returns
        -------
        cache : Tensor
            The created cache tensor.
        """
        if isinstance(readers, (_tensor.Tensor, _tensor.Operation)):
            readers = [readers]
        readers = [t.op if isinstance(t, _tensor.Tensor) else t for t in readers]
        return _ffi_api.ScheduleCacheReadOpaque(self, tensor, scope, readers, suffix)

    def cache_read_opaque_all_readers(self, tensor, scope, suffix = ''):
        """Create a cache read of original tensor for readers.

        This will mutate the body of the readers.
        A new cache stage will be created for the tensor.
        Call this before doing any split/fuse schedule.

        Parameters
        ----------
        tensor : Tensor
            The tensor to be cached.
        scope : str
            The scope of cached

        Returns
        -------
        cache : Tensor
            The created cache tensor.
        """
        return _ffi_api.ScheduleCacheReadOpaqueAllReaders(self, tensor, scope, suffix)

    def cache_write(self, tensor, scope, storage_layout_mode="dense"):
        """Create a cache write of original tensor, before storing into tensor.

        This will mutate the body of the tensor.
        A new cache stage will created before feed into the tensor.

        This function can be used to support data layout transformation.
        If there is a split/fuse/reorder on the data parallel axis of tensor
        before cache_write is called. The intermediate cache stores
        the data in the layout as the iteration order of leave axis.
        The data will be transformed back to the original layout in the original tensor.
        User can further call compute_inline to inline the original layout and keep
        the data stored in the transformed layout.

        Parameters
        ----------
        tensor : Tensor, list or tuple
            The tensors to be feed to. All the tensors must be produced by one computeOp
        scope : str
            The scope of cached

        Returns
        -------
        cache : Tensor
            The created cache tensor.
        """
        return _ffi_api.ScheduleCacheWrite(self, tensor, scope, storage_layout_mode)

    def split_tensor_dimension(self, tensor, dimension, factor):
        """Split tensor dimension to change its data layout

        Parameters
        ----------
        tensor : Tensor
            The tensor whose dimension is to be split
        dimension : int
            The dimension of the tensor to be split
        scope : int
            The split factor


        Returns
        -------
        cache : Tensor
            The created cache tensor.
        """
        return _ffi_api.ScheduleSplitTensorDimension(self, tensor, dimension, factor)


    def fuse_tensor_dimensions(self, tensor, dimension1, dimension2, factor=-1):
        """Split tensor dimension to change its data layout

        Parameters
        ----------
        tensor : Tensor
            The tensor whose dimension is to be split
        dimension : int
            The dimension of the tensor to be split
        scope : int
            The split factor


        Returns
        -------
        cache : Tensor
            The created cache tensor.
        """
        return _ffi_api.ScheduleFuseTensorDimensions(self, tensor, dimension1, dimension2, factor)


    def reorder_tensor_dimensions(self, tensor, dimension1, dimension2):
        """Split tensor dimension to change its data layout

        Parameters
        ----------
        tensor : Tensor
            The tensor whose dimension is to be split
        dimension : int
            The dimension of the tensor to be split
        scope : int
            The split factor


        Returns
        -------
        cache : Tensor
            The created cache tensor.
        """
        return _ffi_api.ScheduleReorderTensorDimensions(self, tensor, dimension1, dimension2)


    def rfactor(self, tensor, axis, factor_axis=0, rfactor_dim = None):
        """ Factor a reduction axis in tensor's schedule to be an explicit axis.

        This will create a new stage that generated the new tensor with axis
        as the first dimension. The tensor's body will be rewritten as a reduction
        over the factored tensor.

        Parameters
        ----------
        tensor : Tensor
            The tensor to be factored.
        axis : IterVar
            The reduction axis in the schedule to be factored.
        factor_axis : int
            The position where the new axis is placed.

        Returns
        -------
        tfactor : Tensor or Array of Tensor
            The created factored tensor.
        """
        factored = _ffi_api.ScheduleRFactor(self, tensor, axis, factor_axis, rfactor_dim)
        return factored[0] if len(factored) == 1 else factored


    def hfuse(self, fuse_tuples):
        ops, ivs = list(zip(*fuse_tuples))
        print(ops, ivs)
        _ffi_api.ScheduleHFuse(self, list(ops), list(ivs))

@tvm._ffi.register_object
class Stage(Object):
    """A Stage represents schedule for one operation."""
    def split(self, parent, factor=None, nparts=None):
        """Split the stage either by factor providing outer scope, or both

        Parameters
        ----------
        parent : IterVar
             The parent iter var.

        factor : Expr, optional
             The splitting factor

        nparts : Expr, optional
             The number of outer parts.

        Returns
        -------
        outer : IterVar
            The outer variable of iteration.

        inner : IterVar
            The inner variable of iteration.
        """
        if nparts is not None:
            if factor is not None:
                raise ValueError("Do not need to provide both outer and nparts")
            outer, inner = _ffi_api.StageSplitByNParts(self, parent, nparts)
        else:
            if factor is None:
                raise ValueError("Either nparts or factor need to be provided")
            outer, inner = _ffi_api.StageSplitByFactor(self, parent, factor)
        return outer, inner

    def fuse(self, *args, padding = -1):
        """Fuse multiple consecutive iteration variables into a single iteration variable.

        fused = fuse(...fuse(fuse(args[0], args[1]), args[2]),..., args[-1])
        The order is from outer to inner.

        Parameters
        ----------
        args : list of IterVars
            Itervars that proceeds each other

        Returns
        -------
        fused : IterVar
            The fused variable of iteration.
        """
        fused = _ffi_api.StageFuse(self, args, padding)
        return fused

    def set_scope(self, scope):
        """Set the thread scope of this stage

        Parameters
        ----------
        scope : str
            The thread scope of this stage
        """
        return _ffi_api.StageSetScope(self, scope)

    def mark_no_sync(self, val = "no_sync"):
        """Mark a tensor so that TVM does consider dependences on it for the
        purposes of barrier insertion.

        This will mutate the body of the readers.
        A new cache stage will be created for the tensor.
        Call this before doing any split/fuse schedule.

        Parameters
        ----------
        tensor : Tensor
            The tensor to be marked.

        """
        _ffi_api.StageMarkNoSync(self, val)

    def mark_relax_storage(self, ):
        """Mark a tensor so that TVM does consider dependences on it for the
        purposes of barrier insertion.

        This will mutate the body of the readers.
        A new cache stage will be created for the tensor.
        Call this before doing any split/fuse schedule.

        Parameters
        ----------
        tensor : Tensor
            The tensor to be marked.

        """
        _ffi_api.StageMarkRelaxStorage(self)

    def mark_no_bounds_check(self):
        """Mark a tensor so that TVM does not generate
        bounds checking for the stage
        """
        _ffi_api.StageMarkNoBoundsCheck(self)

    def mark_no_relax(self, iv):
        """Mark a tensor so that TVM does consider dependences on it for the
        purposes of barrier insertion.

        This will mutate the body of the readers.
        A new cache stage will be created for the tensor.
        Call this before doing any split/fuse schedule.

        Parameters
        ----------
        tensor : Tensor
            The tensor to be marked.

        """
        _ffi_api.StageMarkNoRelax(self, iv)

    def bind(self, ivar, thread_ivar, no_unroll_vthread = False):
        """Bind ivar to thread index thread_ivar

        Parameters
        ----------
        ivar : IterVar
            The iteration to be binded to thread.

        thread_ivar : IterVar
            The thread to be binded.
        """
        _ffi_api.StageBind(self, ivar, thread_ivar)
        if (no_unroll_vthread):
            _ffi_api.StageNoUnrollVThread(self, ivar)

    def env_threads(self, threads):
        """Mark threads to be launched at the outer scope of composed op.

        Parameters
        ----------
        threads : list of threads
            The threads to be launched.
        """
        if isinstance(threads, IterVar):
            threads = [threads]
        _ffi_api.StageEnvThreads(self, threads)

    def set_store_predicate(self, predicate):
        """Set predicate under which store to the array can be performed.

        Use this when there are duplicated threads doing the same store and we only
        need one of them to do the store.

        Parameters
        ----------
        predicate : Expr
            The guard condition fo store.
        """
        _ffi_api.StageSetStorePredicate(self, predicate)

    def compute_at(self, parent, scope):
        """Attach the stage at parent's scope

        Parameters
        ----------
        parent : Stage
            The parent stage

        scope : IterVar
            The loop scope t be attached to.
        """
        _ffi_api.StageComputeAt(self, parent, scope)

    def compute_inline(self):
        """Mark stage as inline

        Parameters
        ----------
        parent : Stage
            The parent stage
        """
        _ffi_api.StageComputeInline(self)

    def compute_root(self):
        """Attach the stage at parent, and mark it as root

        Parameters
        ----------
        parent : Stage
            The parent stage
        """
        _ffi_api.StageComputeRoot(self)

    def reorder(self, *args):
        """reorder the arguments in the specified order.

        Parameters
        ----------
        args : list of IterVar
            The order to be ordered
        """
        _ffi_api.StageReorder(self, args)

    def tile(self, x_parent, y_parent, x_factor, y_factor):
        """ Perform tiling on two dimensions

        The final loop order from outmost to inner most are
        [x_outer, y_outer, x_inner, y_inner]

        Parameters
        ----------
        x_parent : IterVar
            The original x dimension
        y_parent : IterVar
            The original y dimension
        x_factor : Expr
            The stride factor on x axis
        y_factor : Expr
            The stride factor on y axis

        Returns
        -------
        x_outer : IterVar
            Outer axis of x dimension
        y_outer : IterVar
            Outer axis of y dimension
        x_inner : IterVar
            Inner axis of x dimension
        p_y_inner : IterVar
            Inner axis of y dimension
        """
        x_outer, y_outer, x_inner, y_inner = _ffi_api.StageTile(
            self, x_parent, y_parent, x_factor, y_factor)
        return x_outer, y_outer, x_inner, y_inner

    def vectorize(self, var):
        """Vectorize the iteration.

        Parameters
        ----------
        var : IterVar
            The iteration to be vectorize
        """
        _ffi_api.StageVectorize(self, var)

    def tensorize(self, var, tensor_intrin):
        """Tensorize the computation enclosed by var with tensor_intrin

        Parameters
        ----------
        var : IterVar
            The iteration boundary of tensorization.

        tensor_intrin : TensorIntrin
            The tensor intrinsic used for computation.
        """
        _ffi_api.StageTensorize(self, var, tensor_intrin)

    def unroll(self, var):
        """Unroll the iteration.

        Parameters
        ----------
        var : IterVar
            The iteration to be unrolled.
        """
        _ffi_api.StageUnroll(self, var)

    def peel(self, var):
        """Peel the last iteration.

        Parameters
        ----------
        var : IterVar
            The iteration to be peeled.
        """
        _ffi_api.StagePeel(self, var)

    def split_loop(self, var):
        """Split the loop iteration.

        Parameters
        ----------
        var : IterVar
            The iteration to be peeled.
        """
        _ffi_api.StageSplitLoop(self, var)

    def parallel(self, var):
        """Parallelize the iteration.

        Parameters
        ----------
        var : IterVar
            The iteration to be parallelized.
        """
        _ffi_api.StageParallel(self, var)

    def pragma(self, var, pragma_type, pragma_value=None):
        """Annotate the iteration with pragma

        This will translate to a pragma_scope surrounding
        the corresponding loop generated.
        Useful to support experimental features and extensions.

        Parameters
        ----------
        var : IterVar
            The iteration to be anotated

        pragma_type : str
             The pragma string to be annotated

        pragma_value : Expr, optional
             The pragma value to pass along the pragma

        Note
        ----
        Most pragmas are advanced/experimental features
        and may subject to change. List of supported pragmas:

        - **debug_skip_region**

          Force skip the region marked by the axis and turn it into no-op.
          This is useful for debug purposes.

        - **parallel_launch_point**

          Specify to launch parallel threads outside the
          specified iteration loop. By default the threads
          launch at the point of parallel construct.
          This pragma moves the launching point to even outer scope.
          The threads are launched once and reused across multiple
          parallel constructs as BSP style program.

        - **parallel_barrier_when_finish**

          Insert a synchronization barrier between working threads
          after the specified loop iteration finishes.

        - **parallel_stride_pattern**

          Hint parallel loop to execute in strided pattern.
          :code:`for (int i = task_id; i < end; i += num_task)`

        """
        if isinstance(pragma_value, string_types):
            pragma_value = convert(pragma_value)
        _ffi_api.StagePragma(self, var, pragma_type, pragma_value)

    def prefetch(self, tensor, var, offset):
        """Prefetch the specified variable

        Parameters
        ----------
        tensor : Tensor
            The tensor to be prefetched
        var : IterVar
            The loop point at which the prefetching is applied
        offset : Expr
            The number of iterations to be prefetched before actual execution
        """
        _ffi_api.StagePrefetch(self, tensor, var, offset)

    def storage_align(self, axis, factor, offset):
        """Set alignment requirement for specific axis

        This ensures that stride[axis] == k * factor + offset for some k.
        This is useful to set memory layout to for more friendly memory
        access pattern. For example, we can set alignment to be
        factor=2, offset=1 to avoid bank conflict for thread access on
        higher dimension in GPU shared memory.

        Parameters
        ----------
        axis : IterVar
            The axis dimension to be aligned.
        factor : int
            The factor in alignment specification.
        offset : int
            The offset in the alignment specification.
        """
        _ffi_api.StageStorageAlign(self, axis, factor, offset)

    def storage_align_dim(self, dim_idx, factor, offset):
        """Set alignment requirement for specific axis

        This ensures that stride[axis] == k * factor + offset for some k.
        This is useful to set memory layout to for more friendly memory
        access pattern. For example, we can set alignment to be
        factor=2, offset=1 to avoid bank conflict for thread access on
        higher dimension in GPU shared memory.

        Parameters
        ----------
        axis : IterVar
            The axis dimension to be aligned.
        factor : int
            The factor in alignment specification.
        offset : int
            The offset in the alignment specification.
        """
        _ffi_api.StageStorageAlignDim(self, dim_idx, factor, offset)

    def double_buffer(self):
        """Compute the current stage via double buffering.

        This can only be applied to intermediate stage.
        This will double the storage cost of the current stage.
        Can be useful to hide load latency.
        """
        _ffi_api.StageDoubleBuffer(self)

    def opengl(self):
        """The special OpenGL schedule

        Maps each output element to a pixel.
        """
        _ffi_api.StageOpenGL(self)


tvm._ffi._init_api("schedule", __name__)
