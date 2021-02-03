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
"""Abstraction for array data structures."""
from numbers import Integral
import tvm._ffi

from tvm._ffi.base import string_types
from tvm.runtime import Object, convert
from tvm.ir import PrimExpr
from . import _ffi_api
from .expr import Var


@tvm._ffi.register_object
class TensorArray(Object):
    @property
    def ndims(self):
        return len(self.shape)

    @property
    def shape(self):
        return self.__getattr__("shape")

    @property
    def tensor_shape(self):
        return self.__getattr__("tensor_shape")

    @property
    def name(self):
        return self.__getattr__("name")

    @property
    def dtype(self):
        return _ffi_api.TensorArrayGetDType(self)

def decl_region_tensor_array(shape,
                             tensor_shape,
                             dtype=None,
                             name="region_ta"):
    """Declare a new RegionTensorArray.

    Parameters
    ----------
    shape : tuple of Expr
        The shape of the tensor array.

    tensor_shape : tuple of Expr
        The shape of the tensors in the tensor array.

    dtype : str, optional
        The data type of the buffer.

    name : str, optional
        The name of the buffer.

    Returns
    -------
    region_ta : RegionTensorArray
        The created RegionTensoArray
    """
    data = Var(name, "handle")
    return _ffi_api.RegionTensorArray(
        data, dtype, shape, tensor_shape, name)

def decl_pointer_tensor_array(shape,
                              region_ta,
                              name="pointer_ta"):
    """Declare a new PointerTensorArray.

    Parameters
    ----------
    shape : tuple of Expr
        The shape of the buffer.

    region_ta : RegionTensorArray
        The RegionTensorArray corresponding to this PointerTensorArray.

    name : str, optional
        The name of the buffer.

    Returns
    -------
    pointer_ta : PointerTensorArray
        The created PointerTensoArray
    """
    data = Var(name, "handle")
    return _ffi_api.PointerTensorArray(
        data, region_ta, shape, name)


def decl_reshaped_tensor_array(base,
                               shape,
                               tensor_shape,
                               name="region_ta"):
    """Declare a new reshaped RegionTensorArray.

    Parameters
    ----------
    base : RegionTensorArray
        The base RegionTensorArray.

    shape : tuple of Expr
        The shape of the tensor array.

    tensor_shape : tuple of Expr
        The shape of the tensors in the tensor array.

    name : str, optional
        The name of the buffer.

    Returns
    -------
    region_ta : RegionTensorArray
        The created RegionTensoArray
    """
    data = Var(name, "handle")
    return _ffi_api.RegionTensorArrayWithBase(
        data, base.dtype, shape, tensor_shape, name, base)

def lower_tensor_array(declarations, inputs,
                       input_program, target, config, print_body = False):
    return _ffi_api.lower_tensor_arrays(declarations, inputs,
                                        input_program, target, config, print_body)

def lift_to_te(declarations, input_program):
    return _ffi_api.lift_to_te(declarations, input_program)

def check_ta_uses(declarations, input_program):
    return _ffi_api.check_ta_uses(declarations, input_program)

def build_tensor_array(funcs, target, target_host, config):
    return _ffi_api.build_tensor_arrays(funcs, target, target_host, config)
