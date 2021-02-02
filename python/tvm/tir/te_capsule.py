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


@tvm._ffi.register_object
class TECapsule(Object):
    @property
    def name(self):
        return self.__getattr__("name")

    @property
    def inputs(self):
        return list(self.__getattr__("inputs"))

    @property
    def outputs(self):
        return list(self.__getattr__("outputs"))

    @property
    def schedule(self):
        _ffi_api.InitSchedule(self)
        return self.__getattr__("schedule")

    def get_tensor(self, op_name, idx=0):
        return _ffi_api.TECapsuleGetTensor(self, op_name, idx)

        # for t in self.inputs:
        #     if t.op.name == op_name and t.value_index == idx:
        #         return t
        # for t in self.outputs:
        #     if t.op.name == op_name and t.value_index == idx:
        #         return t
        # return None

def create_te_capsule(input_vars,
                      inputs,
                      outputs,
                      tensor_buffer_bounds = {},
                      name="te_capsule"):
    """Declare a new RegionTensorArray.

    Parameters
    ----------
    inputs : List[Tensor]
        The inputs to the TE graph.

    outputs : List[Tensor]
        The outputs of the TE graph.

    tensor_buffer_bounds : Map[Tensor, [Range]]
        For interface tensors (inputs and outputs) in non-global
        memory, bounds to describe how the tensor data is distributed
        across the shared/local memories of the GPU.

    name : str, optional
        The name of the TE graph.

    Returns
    -------
    te_capsule : TECapsule
        The TECapsule encapsulating the TE graph

    """
    return _ffi_api.CreateTECapsule(input_vars, inputs, outputs, tensor_buffer_bounds, name)
