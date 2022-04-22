
from tvm.autotvm.util import get_const_int

def get_cora_axis_length(var):
    return get_const_int(var.dom.extent.func.range.min + var.dom.extent.func.range.extent - 1)