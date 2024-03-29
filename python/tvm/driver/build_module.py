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
"""The build utils in python.

This module provides the functions to transform schedule to
LoweredFunc and compiled Module.
"""
import warnings

import tvm.tir

from tvm.runtime import ndarray
from tvm.ir import container
from tvm.target import codegen, BuildConfig
from tvm.tir import ir_pass
from tvm.tir import Modes
from tvm.tir.stmt import LoweredFunc
from tvm.te import tensor
from tvm.te import schedule
from tvm import target as _target


def get_binds(sch, args, compact=False, binds=None):
    """Internal function to get binds and arg_list given arguments.

    Parameters
    ----------
    args : list of Buffer or Tensor or Var
        The argument lists to the function.

    compact : bool
        If the statement has already bound to a compact buffer.

    binds : dict of :any:`Tensor` to :any:`Buffer`, optional
        Dictionary that maps the Tensor to Buffer which specified the data layout
        requirement of the function. By default, a new compact buffer is created
        for each tensor in the argument.

    Returns
    -------
    binds: dict
        The bind specification

    arg_list: list
        The list of symbolic buffers of arguments.
    """
    binds = {} if binds is None else binds.copy()
    cfg = BuildConfig.current()
    arg_list = []

    def handle_arg(x):
        if isinstance(x, tensor.Tensor):
            any_dim = any(isinstance(i, tvm.tir.Var) for i in x.shape)
            # buffer_type = "auto_broadcast" if any_dim and not compact else ""
            buffer_type = ""
            if isinstance(x.op, tvm.te.ScanOp): sync_type = 1
            else: sync_type = 0
            if x not in binds:
                layout = x.op.output_layout(x.value_index)
                if layout is not None:
                    # print(x, dims, x.shape, type(x.shape))
                    # print('Creating buffer for ' + str(type(x.op)))
                    buf = tvm.tir.decl_buffer(
                        layout,
                        dtype=x.dtype,
                        name=x.name,
                        data_alignment=cfg.data_alignment,
                        offset_factor=cfg.offset_factor,
                        buffer_type=buffer_type, sync_type=sync_type)
                else:
                    buf = tvm.tir.decl_buffer(
                        x.shape,
                        dtype=x.dtype,
                        name=x.name,
                        data_alignment=cfg.data_alignment,
                        offset_factor=cfg.offset_factor,
                        buffer_type=buffer_type, sync_type=sync_type)
                binds[x] = buf
                return buf
            else:
                return binds[x]
        elif isinstance(x, schedule.Buffer):
            return x
        elif isinstance(x, tvm.tir.Var):
            return x
        else:
            raise ValueError("args must be Tensor, Buffer or Var %s" % x)

    for l in args:
        lo = []
        for x in l:
            lo.append(handle_arg(x))
        arg_list.append(lo)
    return binds, arg_list


def form_body(sch, distinct_device, afuns_for):
    """According to the given schedule, form the raw body
    Parameters
    ----------
    sch : tvm.schedule.Schedule
    The given scheduler to form the raw body

    Returns
    -------
    The body formed according to the given schedule
    """
    cfg = BuildConfig.current()
    # normalize schedule first
    sch = sch.normalize()
    # print("[TVM] Made schedule")
    bounds = schedule.InferBound(sch)
    # print("[TVM] Inferred bounds")
    stmt = schedule.ScheduleOps(sch, bounds, False, distinct_device,
                                cfg.fill_in_function_bodies, afuns_for)
    # print("[TVM] Lowered code")
    stmt = ir_pass.InjectPrefetch(stmt)
    return stmt


def lower(sch,
          args,
          target,
          name="default_function",
          binds=None,
          simple_mode=False,
          substitutes=None,
          substitute_after_hfuse=False,
          constraints=[]):
    """Lowering step before build into target.

    Parameters
    ----------
    sch : tvm.schedule.Schedule
        The schedule to be built

    args : list of Buffer or Tensor or Var
        The argument lists to the function.

    name : str, optional
        The name of result function.

    binds : dict of :any:`Tensor` to :any:`Buffer`, optional
        Dictionary that maps the Tensor to Buffer which specified the data layout
        requirement of the function. By default, a new compact buffer is created
        for each tensor in the argument.

    simple_mode : bool, optional
        Whether only output simple and compact statement, this will skip
        LoopPartition, api wrapper generation and Unrolling.

    constraints : list of exprs, optional
        Constraints on input variables that might help simplify some
        expressions/get rid of some conditionals

    Returns
    -------
    f : LoweredFunc or Stmt
       The result function, if with_api_wrapper=False
       Then the Stmt before make api is returned.

    """
    cfg = BuildConfig.current()
    add_lower_pass = cfg.add_lower_pass if cfg.add_lower_pass else []
    if cfg.dump_pass_ir:
        add_lower_pass = BuildConfig._dump_ir.decorate_custompass(add_lower_pass)
    lower_phase0 = [x[1] for x in add_lower_pass if x[0] == 0]
    lower_phase1 = [x[1] for x in add_lower_pass if x[0] == 1]
    lower_phase2 = [x[1] for x in add_lower_pass if x[0] == 2]
    lower_phase3 = [x[1] for x in add_lower_pass if x[0] > 2]

    afuns_for = []
    for arg in [item for sublist in args for item in sublist]:
        if isinstance(arg, tvm.tir.Buffer):
            afuns_for.append(arg)

    if binds:
        for buf in binds.values():
            afuns_for.append(buf)

    # Phase 0
    if isinstance(sch, schedule.Schedule):
        stmt = form_body(sch, target != "c" and target != "llvm", afuns_for)
    # exit(0)

    for f in lower_phase0:
        stmt = f(stmt)

    compact = ir_pass.VerifyCompactBuffer(stmt)
    binds, arg_list = get_binds(sch, args, compact, binds)

    # Phase 1
    stmt = ir_pass.RewriteForTensorCore(stmt, sch, binds)
    # if simple_mode: print(stmt)
    # exit(0)
    stmt = ir_pass.StorageFlatten(stmt, binds, 64, cfg.instrument_bound_checkers)
    # stmt = ir_pass.CanonicalSimplify(stmt)
    for f in lower_phase1:
        stmt = f(stmt)

    # print("[TVM] Phase 2")

    # Phase 2
    stmt = ir_pass.RemoveRedundantIfs(stmt, constraints)
    # if not simple_mode:
        # stmt = ir_pass.LoopPartition(stmt, cfg.partition_const_loop)

    stmt = ir_pass.RemoveLikelyTags(stmt)

    # print(stmt)
    # exit(0)
    stmt = ir_pass.Simplify(stmt)
    # print(stmt)
    # exit(0)
    if cfg.disable_vectorize:
        stmt = ir_pass.SkipVectorize(stmt)
    else:
        stmt = ir_pass.VectorizeLoop(stmt)
    stmt = ir_pass.InjectVirtualThread(stmt)
    stmt = ir_pass.InjectDoubleBuffer(stmt, cfg.double_buffer_split_loop)
    stmt = ir_pass.StorageRewrite(stmt)
    stmt = ir_pass.UnrollLoop(
        stmt,
        cfg.auto_unroll_max_step,
        cfg.auto_unroll_max_depth,
        cfg.auto_unroll_max_extent,
        cfg.unroll_explicit)
    for f in lower_phase2:
        stmt = f(stmt)

    # Phase 3
    stmt = ir_pass.Simplify(stmt)
    stmt = ir_pass.RemoveNoOp(stmt)
    if not cfg.disable_select_rewriting:
        stmt = ir_pass.RewriteUnsafeSelect(stmt)
    for f in lower_phase3:
        stmt = f(stmt)

    # Instrument BoundCheckers
    if cfg.instrument_bound_checkers:
        stmt = ir_pass.InstrumentBoundCheckers(stmt)

    # Adding this pass here results in incorrect optimizations being
    # applied. Disabling for now
    # stmt = ir_pass.HoistIfThenElse(stmt)
    stmt = ir_pass.ExpandIntrinsicITE(stmt)

    if substitutes and not substitute_after_hfuse:
        print('Lowering substitute')
        stmt = ir_pass.SubstituteThreadVars(stmt, substitutes[0], substitutes[1])
        # print(stmt)
        # exit(0)

    if simple_mode:
        try:
            arg_list = [list(dict.fromkeys(l)) for l in arg_list]
            # ret = ir_pass.MakeAPI(stmt, name, arg_list[0], arg_list[1], 0, cfg.restricted_func, True)
            # print(ret.function.body)
            # exit(0)
        except:
            print(stmt)
            raise
        return stmt

    # Remove duplicates
    arg_list = [list(dict.fromkeys(l)) for l in arg_list]
    if cfg.prep_code_mode == "with_prep_code":
        make_api_result = ir_pass.MakeAPIWithPrepCode(stmt, name, arg_list[0], arg_list[1], 0, cfg.restricted_func)
    elif cfg.prep_code_mode == "no_prep_code":
        make_api_result = ir_pass.MakeAPINoPrepCode(stmt, name, arg_list[0], arg_list[1], 0, cfg.restricted_func)
    elif cfg.prep_code_mode == "only_prep_code":
        make_api_result = ir_pass.MakeAPIOnlyPrepCode(stmt, name, arg_list[0], arg_list[1], 0, cfg.restricted_func)
    else:
        raise ValueError("No such prep_code_mode: " + prep_code_mode)

    return make_api_result

def _build_for_device(flist, target, target_host, constraints=[],
                      cuda_syncs=None, substitute_after_hfuse=False,
                      substitutes=None):
    """Build the lowered functions for a device with the given compilation
    target.

    Parameters
    ----------
    flist : list of LoweredFunc
        The schedule to be built.

    target : str or :any:`tvm.target.Target`
        The target and option of the compilation.

    target_host : str or :any:`tvm.target.Target`
        The host compilation target.

    Returns
    -------
    fhost : list of LoweredFunc
        A list of lowered functions for the host.

    mdev : tvm.module
        A module that contains device code.
    """
    target = _target.create(target)
    device_type = ndarray.context(target.target_name, 0).device_type
    fhost = []
    fdevice = []
    for func in flist:
        if not ir_pass.VerifyMemory(func, device_type):
            raise ValueError(
                "Direct host side access to device memory is detected in %s. "
                "Did you forget to bind?" % func.name)

        if func.func_type == LoweredFunc.MixedFunc:
            if BuildConfig.current().detect_global_barrier:
                func = ir_pass.ThreadSync(func, "global", target.target_name)
                # print(func.body)
            func = ir_pass.ThreadSync(func, "shared", target.target_name)
            func = ir_pass.ThreadSync(func, "warp", target.target_name)
            func = ir_pass.CreateEnvLoopsForFunc(func, target.target_name)
            func = ir_pass.InferFragment(func)
            warp_size = target.thread_warp_size
            func = ir_pass.LowerThreadAllreduce(func, warp_size, target.target_name)
            func = ir_pass.PeelLoop(func)
            cuda_syncs = "" if cuda_syncs == None else cuda_syncs
            ############################################################
            func = ir_pass.RemoveProducerConsumerNodes(func)
            func = ir_pass.BetterHoistIfThenElse(func, target.target_name, constraints)
            # print(func.body)
            # exit(0)
            func = ir_pass.HorizontalFuse(func)
            if substitutes and substitute_after_hfuse:
                print('Building substitute')
                func = ir_pass.SubstituteThreadVarsFunc(func, substitutes[0], substitutes[1])
                # print(func.body)
            # exit(0)
            ############################################################
            fsplits = list(ir_pass.SplitHostDevice(func, cuda_syncs))
            fhost.append(fsplits[0])
            for x in fsplits[1:]:
                fdevice.append(x)
        elif func.func_type == LoweredFunc.HostFunc:
            fhost.append(func)
        elif func.func_type == LoweredFunc.DeviceFunc:
            fdevice.append(func)
        else:
            raise ValueError("unknown function type %d" % func.func_type)

    for i, func in enumerate(fdevice):
        warp_size = target.thread_warp_size
        fdevice[i] = ir_pass.LowerWarpMemory(func, warp_size)

    if "gpu" in target.keys and not fdevice:
        warnings.warn(
            "Specified target %s, but cannot find device code, did you do "
            "bind?" % target)

    fhost = [ir_pass.BindDeviceType(x, device_type) for x in fhost]
    fhost = [ir_pass.LowerTVMBuiltin(x) for x in fhost]

    if device_type == ndarray.cpu(0).device_type and target_host == target:
        assert not fdevice

    target_host = _target.create(target_host)
    fdevice = [ir_pass.LowerDeviceStorageAccessInfo(x) for x in fdevice]
    fhost = [ir_pass.LowerDeviceStorageAccessInfo(x) for x in fhost]
    # print("# DEVICE ##############################\n", fdevice[0].body)
    fdevice = [ir_pass.LowerIntrin(x, target.target_name) for x in fdevice]
    # print("# DEVICE ##############################\n", fdevice[0].body)
    # exit(0)
    fhost = [ir_pass.LowerIntrin(x, target_host.target_name) for x in fhost]
    fhost = [ir_pass.CombineContextCall(x) for x in fhost]

    # fdevice = [ir_pass.BetterHoistIfThenElse(x, target.target_name, constraints) for x in fdevice]
    # if len(fdevice) == 0:
        # fhost = [ir_pass.BetterHoistIfThenElse(x, target.target_name, constraints) for x in fhost]
    # print("# DEVICE ##############################\n", fdevice[0].body)
    # exit(0)
    cfg = BuildConfig.current()
    if cfg.hoist_loads:
        # print('Hoisting')
        fdevice = [ir_pass.HoistLoads(x) for x in fdevice]
    # print("# HOST ##############################\n", fhost[0].body)
    # print("# DEVICE ##############################\n", fdevice[0].body)
    # exit(0)
    mdev = codegen.build_module(fdevice, str(target)) if fdevice else None

    return fhost, mdev


def build(inputs,
          args=None,
          target=None,
          target_host=None,
          name="default_function",
          binds=None,
          substitutes=None,
          substitute_after_hfuse=False,
          constraints=[],
          cuda_syncs=None):
    """Build a function with arguments as signature. Code will be generated
    for devices coupled with target information.

    Parameters
    ----------
    inputs : tvm.Schedule, LoweredFunc, or dict of target to LoweredFunc list
        The schedule to be built

    args : list of Buffer or Tensor or Var, optional
        The argument lists to the function.

    target : str or :any:`tvm.target.Target`, optional
        The target and option of the compilation.

    target_host : str or :any:`tvm.target.Target` optional
        Host compilation target, if target is device.
        When TVM compiles device specific program such as CUDA,
        we also need host(CPU) side code to interact with the driver
        setup the dimensions and parameters correctly.
        target_host is used to specify the host side codegen target.
        By default, llvm is used if it is enabled,
        otherwise a stackvm intepreter is used.

    name : str, optional
        The name of result function.

    binds : dict, optional
        Dictionary that maps the binding of symbolic buffer to Tensor.
        By default, a new buffer is created for each tensor in the argument.

    Returns
    -------
    ret : tvm.module
        A module that combines both host and device code.

    Examples
    ________
    There are two typical example uses of this function depending on the type
    of the argument `inputs`:
    1. it is a list of lowered functions:

    .. code-block:: python

        n = 2
        A = tvm.placeholder((n,), name='A')
        B = tvm.placeholder((n,), name='B')
        C = tvm.compute(A.shape, lambda *i: A(*i) + B(*i), name='C')
        s = tvm.create_schedule(C.op)
        f = tvm.lower(s, [A, B, C], name="test_add")
        m = tvm.build(f, target="llvm")

    2. it is a dict of compilation target to list of lowered functions:

    .. code-block:: python

        n = 2
        A = tvm.placeholder((n,), name='A')
        B = tvm.placeholder((n,), name='B')
        C = tvm.compute(A.shape, lambda *i: A(*i) + B(*i), name='C')
        s1 = tvm.create_schedule(C.op)
        with tvm.target.cuda() as cuda_tgt:
          s2 = topi.cuda.schedule_injective(cuda_tgt, [C])
          f1 = tvm.lower(s1, [A, B, C], name="test_add1")
          f2 = tvm.lower(s2, [A, B, C], name="test_add2")
          m = tvm.build({"llvm": [f1], "cuda": [f2]}, target_host="llvm")

    Note
    ----
    See the note on :any:`tvm.target` on target string format.
    """
    intermediate_buffers = None
    if isinstance(inputs, schedule.Schedule):
        if args is None:
            raise ValueError("args must be given for build from schedule")
        make_api_result = lower(inputs, args, target,
                                name=name,
                                binds=binds,
                                substitutes=substitutes,
                                substitute_after_hfuse=substitute_after_hfuse,
                                constraints=constraints)
        flist = make_api_result.function
        # print(flist.body)
        # exit(0)
        intermediate_buffers = (make_api_result.host_intermediate_buffers,
                                make_api_result.device_intermediate_buffers)
        if isinstance(flist, LoweredFunc):
            flist = [flist]
    elif isinstance(inputs, LoweredFunc):
        if args:
            raise ValueError("args must be done when build from LoweredFunc.")
        flist = [inputs]
    elif isinstance(inputs, (list, tuple, container.Array)):
        flist = inputs
    elif not isinstance(inputs, (dict, container.Map)):
        raise ValueError("inputs must be Schedule, LoweredFunc, list of "
                         "LoweredFunc, or dict of target to list of "
                         "LoweredFunc.")

    if not isinstance(inputs, (dict, container.Map)):
        target = _target.Target.current() if target is None else target
        target = target if target else "llvm"
        target_flist = {target: flist}
    else:
        target_flist = inputs

    for tar, flist in target_flist.items():
        if not isinstance(tar, (str, _target.Target)):
            raise ValueError("The key of inputs must be str or "
                             "_target.Target when inputs is dict.")
        fname_set = set()
        for x in flist:
            if not isinstance(x, LoweredFunc):
                raise ValueError("inputs must be Schedule, LoweredFunc, list "
                                 "of LoweredFunc, or dict of str to list of "
                                 "LoweredFunc.")
            if x.name in fname_set:
                raise ValueError("Duplicate function name %s" % x.name)
            fname_set.add(x.name)

    if not target_host:
        for tar, _ in target_flist.items():
            tar = _target.create(tar)
            device_type = ndarray.context(tar.target_name, 0).device_type
            if device_type == ndarray.cpu(0).device_type:
                target_host = tar
                break
    if not target_host:
        target_host = "llvm" if tvm.runtime.enabled("llvm") else "stackvm"

    fhost_all = []
    device_modules = []
    for tar, flist in target_flist.items():
        fhost, mdev = _build_for_device(flist, tar, target_host,
                                        constraints=constraints,
                                        cuda_syncs=cuda_syncs,
                                        substitutes=substitutes,
                                        substitute_after_hfuse=substitute_after_hfuse)
        # Save the current lowered functions of the host and the device module.
        fhost_all += fhost
        device_modules.append(mdev)

    # Generate a unified host module.
    mhost = codegen.build_module(fhost_all, str(target_host))

    # Import all modules.
    for mdev in device_modules:
        if mdev:
            mhost.import_module(mdev)
    return mhost, intermediate_buffers
