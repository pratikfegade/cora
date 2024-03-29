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
# pylint: disable=redefined-builtin, invalid-name
"""Operators used in TIR expression."""
import tvm._ffi
from tvm.runtime import convert, const
from tvm.ir import Array

from .buffer import Buffer
from .expr import Call, Var, CommReducer
from . import _ffi_api


def _pack_buffer(buf):
    """Build intrinsics that packs the buffer.
    """
    assert buf.shape
    shape = Call("handle", "tvm_stack_make_shape", buf.shape.dense_shape(),
                 Call.Intrinsic, None, 0)
    strides = Call("handle", "tvm_stack_make_shape", buf.strides,
                   Call.Intrinsic, None, 0) if buf.strides else 0
    pack_args = [buf.data,
                 shape,
                 strides,
                 len(buf.shape.dense_shape()),
                 const(0, dtype=buf.dtype),
                 buf.elem_offset]
    return Call("handle", "tvm_stack_make_array",
                pack_args, Call.Intrinsic, None, 0)

def call_packed(*args):
    """Build expression by call an external packed function.

    The argument to packed function can be Expr or Buffer.
    The argument is the corresponding POD type when Expr is presented.

    When the argument is Buffer, the corresponding PackedFunc
    will recieve an TVMArrayHandle whose content is valid during the callback period.
    If the PackedFunc is a python callback, then the corresponding argument is NDArray.

    Parameters
    ----------
    args : list of Expr or Buffer.
        Positional arguments.

    Returns
    -------
    call : PrimExpr
        The call expression.

    See Also
    --------
    tvm.extern : Create tensor with extern function call.
    """
    call_args = [_pack_buffer(x) if isinstance(x, Buffer) else x for x in args]
    return Call(
        "int32", "tvm_call_packed", call_args, Call.Intrinsic, None, 0)


def call_pure_intrin(dtype, func_name, *args):
    """Build expression by calling a pure intrinsic function.

    Intrinsics can be overloaded with multiple data types via
    the intrinsic translation rule.

    Parameters
    ----------
    dtype : str
        The data type of the result.

    func_name: str
        The intrinsic function name.

    args : list
        Positional arguments.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    args = convert(args)
    return Call(
        dtype, func_name, convert(args), Call.PureIntrinsic, None, 0)


def call_intrin(dtype, func_name, *args):
    """Build expression by calling an intrinsic function.

    Intrinsics can be overloaded with multiple data types via
    the intrinsic translation rule.

    Parameters
    ----------
    dtype : str
        The data type of the result.

    func_name: str
        The intrinsic function name.

    args : list
        Positional arguments.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    args = convert(args)
    return Call(
        dtype, func_name, convert(args), Call.Intrinsic, None, 0)


def call_pure_extern(dtype, func_name, *args):
    """Build expression by calling a pure extern function.

    Parameters
    ----------
    dtype : str
        The data type of the result.

    func_name: str
        The extern function name.

    args : list
        Positional arguments.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return Call(
        dtype, func_name, convert(args), Call.PureExtern, None, 0)


def call_extern(dtype, func_name, *args):
    """Build expression by calling a extern function.

    Parameters
    ----------
    dtype : str
        The data type of the result.

    func_name: str
        The extern function name.

    args : list
        Positional arguments.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return Call(
        dtype, func_name, convert(args), Call.Extern, None, 0)


def call_llvm_intrin(dtype, name, *args):
    """Build expression by calling an llvm intrinsic function

    Parameters
    ----------
    dtype : str
       The data type of the result.

    name : str
       The name of the llvm intrinsic function.

    args : list
       Poistional arguments.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    # pylint: disable=import-outside-toplevel
    from tvm.target import codegen
    llvm_id = codegen.llvm_lookup_intrinsic_id(name)
    assert llvm_id != 0, "%s is not an LLVM intrinsic" % name
    return call_pure_intrin(dtype, 'llvm_intrin', tvm.const(llvm_id, 'uint32'), *args)


def any(*args):
    """Create a new experssion of the union of all conditions in the arguments

    Parameters
    ----------
    args : list
        List of symbolic boolean expressions

    Returns
    -------
    expr: Expr
        Expression
    """
    if not args:
        raise ValueError("Any must take at least 1 argument")
    if len(args) == 1:
        return args[0]
    ret = _ffi_api._OpOr(args[0], args[1])
    for i in range(2, len(args)):
        ret = _ffi_api._OpOr(ret, args[i])
    return ret


def all(*args):
    """Create a new experssion of the intersection of all conditions in the
      arguments

    Parameters
    ----------
    args : list
        List of symbolic boolean expressions

    Returns
    -------
    expr: Expr
        Expression
    """
    if not args:
        raise ValueError("Any must take at least 1 argument")
    if len(args) == 1:
        return args[0]
    ret = _ffi_api._OpAnd(args[0], args[1])
    for i in range(2, len(args)):
        ret = _ffi_api._OpAnd(ret, args[i])
    return ret


@tvm._ffi.register_func("tvm.default_trace_action")
def _tvm_default_trace_action(*args):
    print(list(args))

def trace(args, trace_action="tvm.default_trace_action"):
    """Trace tensor data at the runtime.

    The trace function allows to trace specific tensor at the
    runtime. The tracing value should come as last argument.
    The trace action should be specified, by default
    tvm.default_trace_action is used.

    Parameters
    ----------
    args : list of Expr or Buffers.
        Positional arguments.

    trace_action : str.
        The name of the trace action.

    Returns
    -------
    call : PrimExpr
        The call expression.

    See Also
    --------
    tvm.tir.call_packed : Creates packed function.
    """
    if not isinstance(args, list):
        raise Exception("tvm.trace consumes the args as list type")
    call_args = [_pack_buffer(x) if isinstance(x, Buffer) else x for x in args]
    call_args.insert(0, trace_action)
    return tvm.tir.Call(
        args[-1].dtype, "tvm_call_trace_packed", call_args, tvm.tir.Call.Intrinsic, None, 0)



def min_value(dtype):
    """minimum value of dtype

    Parameters
    ----------
    dtype : str
        The data type.

    Returns
    -------
    value : tvm.Expr
        The minimum value of dtype.
    """
    return _ffi_api.min_value(dtype)


def max_value(dtype):
    """maximum value of dtype

    Parameters
    ----------
    dtype : str
        The data type.

    Returns
    -------
    value : tvm.Expr
        The maximum value of dtype.
    """
    return _ffi_api.max_value(dtype)


def exp(x):
    """Take exponetial of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return call_pure_intrin(x.dtype, "exp", x)


def fast_exp(x):
    """Take exponetial of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return call_pure_intrin(x.dtype, "fast_exp", x)


def erf(x):
    """Take gauss error function of the input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return call_pure_intrin(x.dtype, "erf", x)


def tanh(x):
    """Take hyperbolic tanh of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return call_pure_intrin(x.dtype, "tanh", x)


def fast_tanh(x):
    """Take hyperbolic tanh of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    def make_const(dtype, val): return const(val, dtype)

    # Clamp the inputs to the range [-9, 9] since anything outside
    # this range is +/-1.0f in single-precision.
    x = tvm.tir.Max(tvm.tir.Min(x, make_const(x.dtype, 9.0)), make_const(x.dtype, -9.0));

    # The monomial coefficients of the numerator polynomial (odd).
    alpha_1 = make_const(x.dtype, 4.89352455891786e-03);
    alpha_3 = make_const(x.dtype, 6.37261928875436e-04);
    alpha_5 = make_const(x.dtype, 1.48572235717979e-05);
    alpha_7 = make_const(x.dtype, 5.12229709037114e-08);
    alpha_9 = make_const(x.dtype, -8.60467152213735e-11);
    alpha_11 = make_const(x.dtype, 2.00018790482477e-13);
    alpha_13 = make_const(x.dtype, -2.76076847742355e-16);

    # The monomial coefficients of the denominator polynomial (even).
    beta_0 = make_const(x.dtype, 4.89352518554385e-03);
    beta_2 = make_const(x.dtype, 2.26843463243900e-03);
    beta_4 = make_const(x.dtype, 1.18534705686654e-04);
    beta_6 = make_const(x.dtype, 1.19825839466702e-06);

    x2 = x * x;
    p = x2 * alpha_13 + alpha_11;
    p = x2 * p + alpha_9;
    p = x2 * p + alpha_7;
    p = x2 * p + alpha_5;
    p = x2 * p + alpha_3;
    p = x2 * p + alpha_1;
    p = x * p;

    q = x2 * beta_6 + beta_4;
    q = x2 * q + beta_2;
    q = x2 * q + beta_0;
    return (p / q);

    # return call_pure_intrin(x.dtype, "fast_tanh", x)


def fast_sigmoid(inp):
    """Quick function to get sigmoid

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    def make_const(dtype, val): return const(val, dtype)

    x_hi = make_const("float32", 88.3762626647950);
    x_lo = make_const("float32", -88.3762626647949);
    log2e = make_const("float32", 1.44269504088896341);
    ln2 = make_const("float32", 0.6931471805599453);
    p = [make_const("float32", 1.9875691500E-4),
         make_const("float32", 1.3981999507E-3),
         make_const("float32", 8.3334519073E-3),
         make_const("float32", 4.1665795894E-2),
         make_const("float32", 1.6666665459E-1),
         make_const("float32", 5.0000001201E-1)];
    one = make_const("float32", 1.0);
    one_half = make_const("float32", 0.5);
    b = make_const("float32", 127.0);

    # clamp x
    x = tvm.tir.Max(tvm.tir.Min(inp, x_hi), x_lo)
    # integer part
    # n = tir::CallNode::make(x.dtype, "floor", {x * log2e + one_half}, tir::CallNode::PureIntrinsic);
    n = tvm.tir.Call(x.dtype, "floor", [x * log2e + one_half], tvm.tir.Call.PureIntrinsic, None, 0)
    # fractional part
    f = x - n * ln2;
    y = (((((p[0] * f + p[1]) * f + p[2]) * f + p[3]) * f + p[4]) * f + p[5]) * f * f + f + one;
    # Return 2^m * exp(r).
    ef = tvm.tir.Call("float32", "reinterpret", [tvm.tir.Cast("int32", n + b) << 23], tvm.tir.Call.PureIntrinsic, None, 0)
    # ef = tvm::reinterpret(DataType::Float(32), ::tvm::cast(DataType::Int(32), n + b) << 23);
    exp = tvm.tir.Max(ef * y, inp);  # NOLINT(*)

    return 1.0 / (1.0 - exp)
    # return call_pure_intrin(x.dtype, "fast_sigmoid", x)


def sigmoid(x):
    """Quick function to get sigmoid

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return call_pure_intrin(x.dtype, "sigmoid", x)


def log(x):
    """Take log of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return call_pure_intrin(x.dtype, "log", x)

def cos(x):
    """Take cos of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return call_pure_intrin(x.dtype, "cos", x)

def sin(x):
    """Take sin of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return call_pure_intrin(x.dtype, "sin", x)

def atan(x):
    """Take atan of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return call_pure_intrin(x.dtype, "atan", x)

def sqrt(x):
    """Take square root of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return call_pure_intrin(x.dtype, "sqrt", x)


def rsqrt(x):
    """Take reciprocal of square root of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return call_pure_intrin(x.dtype, "rsqrt", x)


def floor(x):
    """Take floor of float input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _ffi_api.floor(x)


def ceil(x):
    """Take ceil of float input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _ffi_api.ceil(x)


def trunc(x):
    """Get truncated value of the input.

    The truncated value of the scalar x is the
    nearest integer i which is closer to zero than x is.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _ffi_api.trunc(x)


def abs(x):
    """Get absolute value of the input element-wise.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _ffi_api.abs(x)


def round(x):
    """Round elements of the array to the nearest integer.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _ffi_api.round(x)


def nearbyint(x):
    """Round elements of the array to the nearest integer.
    This intrinsic uses llvm.nearbyint instead of llvm.round
    which is faster but will results different from tvm.round.
    Notably nearbyint rounds according to the rounding mode,
    whereas tvm.round (llvm.round) ignores that.
    For differences between the two see:
    https://en.cppreference.com/w/cpp/numeric/math/round
    https://en.cppreference.com/w/cpp/numeric/math/nearbyint

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _ffi_api.nearbyint(x)


def isnan(x):
    """Check if input value is Nan.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _ffi_api.isnan(x)


def power(x, y):
    """x power y

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    y : PrimExpr
        The exponent

    Returns
    -------
    z : PrimExpr
        The result.
    """
    return _ffi_api._OpPow(convert(x), convert(y))


def popcount(x):
    """Count the number of set bits in input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return call_pure_intrin(x.dtype, "popcount", x)

def fmod(x, y):
    """Return the remainder of x divided by y with the same sign as x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.
    y : PrimExpr
        Input argument.

    Returns
    -------
    z : PrimExpr
        The result.
    """
    return call_pure_intrin(x.dtype, "fmod", x, y)


def if_then_else(cond, t, f):
    """Conditional selection expression.

    Parameters
    ----------
    cond : PrimExpr
        The condition

    t : PrimExpr
        The result expression if cond is true.

    f : PrimExpr
        The result expression if cond is false.

    Returns
    -------
    result : Node
        The result of conditional expression.

    Note
    ----
    Unlike Select, if_then_else will not execute
    the branch that does not satisfy the condition.
    You can use it to guard against out of bound access.
    Unlike Select, if_then_else cannot be vectorized
    if some lanes in the vector have different conditions.
    """
    return _ffi_api._OpIfThenElse(convert(cond), convert(t), convert(f))


def div(a, b):
    """Compute a / b as in C/C++ semantics.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand, known to be non-negative.

    b : PrimExpr
        The right hand operand, known to be non-negative.

    Returns
    -------
    res : PrimExpr
        The result expression.
    Note
    ----
    When operands are integers, returns truncdiv(a, b).
    """
    return _ffi_api._OpDiv(a, b)


def indexdiv(a, b):
    """Compute floor(a / b) where a and b are non-negative.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand, known to be non-negative.

    b : PrimExpr
        The right hand operand, known to be non-negative.

    Returns
    -------
    res : PrimExpr
        The result expression.

    Note
    ----
    Use this function to split non-negative indices.
    This function may take advantage of operands'
    non-negativeness.
    """
    return _ffi_api._OpIndexDiv(a, b)


def indexmod(a, b):
    """Compute the remainder of indexdiv. a and b are non-negative.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand, known to be non-negative.

    b : PrimExpr
        The right hand operand, known to be non-negative.

    Returns
    -------
    res : PrimExpr
        The result expression.

    Note
    ----
    Use this function to split non-negative indices.
    This function may take advantage of operands'
    non-negativeness.
    """
    return _ffi_api._OpIndexMod(a, b)


def truncdiv(a, b):
    """Compute the truncdiv of two expressions.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand

    b : PrimExpr
        The right hand operand

    Returns
    -------
    res : PrimExpr
        The result expression.

    Note
    ----
    This is the default integer division behavior in C.
    """
    return _ffi_api._OpTruncDiv(a, b)


def truncmod(a, b):
    """Compute the truncmod of two expressions.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand

    b : PrimExpr
        The right hand operand

    Returns
    -------
    res : PrimExpr
        The result expression.

    Note
    ----
    This is the default integer division behavior in C.
    """
    return _ffi_api._OpTruncMod(a, b)


def floordiv(a, b):
    """Compute the floordiv of two expressions.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand

    b : PrimExpr
        The right hand operand

    Returns
    -------
    res : PrimExpr
        The result expression.
    """
    return _ffi_api._OpFloorDiv(a, b)


def floormod(a, b):
    """Compute the floormod of two expressions.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand

    b : PrimExpr
        The right hand operand

    Returns
    -------
    res : PrimExpr
        The result expression.
    """
    return _ffi_api._OpFloorMod(a, b)


def comm_reducer(fcombine, fidentity, name="reduce"):
    """Create a commutative reducer for reduction.

    Parameters
    ----------
    fcombine : function(Expr -> Expr -> Expr)
        A binary function which takes two Expr as input to return a Expr.

    fidentity : function(str -> Expr)
        A function which takes a type string as input to return a const Expr.

    Returns
    -------
    reducer : function
        A function which creates a reduce expression over axis.
        There are two ways to use it:

        1. accept (expr, axis, where) to produce an Reduce Expr on
           specified axis;
        2. simply use it with multiple Exprs.

    Example
    -------
    .. code-block:: python

        n = tvm.var("n")
        m = tvm.var("m")
        mysum = tvm.comm_reducer(lambda x, y: x+y,
            lambda t: tvm.const(0, dtype=t), name="mysum")
        A = tvm.placeholder((n, m), name="A")
        k = tvm.reduce_axis((0, m), name="k")
        B = tvm.compute((n,), lambda i: mysum(A[i, k], axis=k), name="B")
    """
    def _reduce_directly(*args):
        num = len(args)
        # process `where` is None
        if num == 3 and args[2] is None:
            num = 2
        res = args[0]
        for i in range(num-1):
            res = fcombine(res, args[i+1])
        return res

    def _make_reduce(expr, axis, dimensions=[], where=None):
        code = fcombine.__code__
        assert fcombine.__code__.co_argcount == 2
        expr = convert(expr)
        if isinstance(expr, Array):
            size = len(expr)
            larr = []
            rarr = []
            dtypes = []
            for i in range(size):
                dtype = expr[i].dtype
                dtypes.append(dtype)
                lname = code.co_varnames[0] + "_" + str(i)
                larr.append(Var(lname, dtype))
                rname = code.co_varnames[1] + "_" + str(i)
                rarr.append(Var(rname, dtype))
            lhs = convert(larr)
            rhs = convert(rarr)
            result = fcombine(lhs, rhs)
            id_elem = fidentity(*dtypes)
        else:
            assert isinstance(expr, tvm.ir.PrimExpr)
            size = 1
            dtype = expr.dtype
            lvar = Var(code.co_varnames[0], dtype)
            rvar = Var(code.co_varnames[1], dtype)
            result = [fcombine(lvar, rvar)]
            id_elem = [fidentity(dtype)]
            lhs = convert([lvar])
            rhs = convert([rvar])
            expr = convert([expr])
        result = convert(result)
        id_elem = convert(id_elem)
        combiner = CommReducer(lhs, rhs, result, id_elem)
        axis = convert(axis if isinstance(axis, (list, tuple)) else [axis])
        dimensions = dimensions if isinstance(dimensions, (list, tuple)) else [dimensions]
        if where is None:
            where = convert(True)
        outputs = tuple(tvm.tir.Reduce(combiner, expr, axis, where, i, dimensions)
                        for i in range(size))
        return outputs[0] if size == 1 else outputs

    # pylint: disable=keyword-arg-before-vararg
    def reducer(expr, axis, dimensions=[], where=None, *args):
        if isinstance(axis, (tvm.tir.IterVar, list, tuple)):
            assert not args
            return _make_reduce(expr, axis, dimensions, where)
        if where is None:
            assert not dimensions
            assert not args
            return _reduce_directly(expr, axis)
        return _reduce_directly(expr, axis, where, *args)

    doc_str = """Create a {0} expression over axis.

              Parameters
              ----------
              expr : PrimExpr
                  The source expression.
              axis : IterVar
                  The reduction IterVar axis
              where : optional, Expr
                  Filtering predicate of the reduction.
              Returns
              -------
              value : PrimExpr
                  The result value.

              Example
              -------
              .. code-block:: python

                m = tvm.var("m")
                n = tvm.var("n")
                A = tvm.placeholder((m, n), name="A")
                k = tvm.reduce_axis((0, n), name="k")

                # there are two way to use this {0} reducer:
                # mode 1, accept (expr, axis, where) to produce an Reduce Expr
                B = tvm.compute((m,), lambda i: tvm.{0}(A[i, k], axis=k), name="B")

                # mode 2, simply use it with multiple Exprs:
                {0}_res = tvm.{0}(m, n)
              """
    reducer.__doc__ = doc_str.format(name)
    return reducer

# pylint: disable=unnecessary-lambda
sum = comm_reducer(lambda x, y: x+y, lambda t: const(0, dtype=t), name="sum")
min = comm_reducer(lambda x, y: _ffi_api._OpMin(x, y), max_value, name="min")
max = comm_reducer(lambda x, y: _ffi_api._OpMax(x, y), min_value, name="max")
