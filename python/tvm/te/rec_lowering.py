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
from tvm.tir import IterVar, Buffer

from . import tensor as _tensor
from . import _ffi_api

class RecVars:
    def __init__(self):
        self.num_nodes_var = tvm.tir.Var('num_nodes', 'int32')
        self.num_batches_var = tvm.tir.Var('num_batches', 'int32')
        self.max_batch_len_var = tvm.tir.Var('max_batch_len', 'int32')
        self.max_child_num_var = tvm.tir.Var('max_child_num', 'int32')
        self.max_int_idx_var = tvm.tir.Var('max_int_idx', 'int32')

def lower_dyn_batch(ops, rec_vars, leaf_specialization):
    return _ffi_api.LowerDynamicBatching(ops, rec_vars.num_nodes_var,
                                         rec_vars.num_batches_var,
                                         rec_vars.max_batch_len_var,
                                         rec_vars.max_child_num_var,
                                         rec_vars.max_int_idx_var,
                                         leaf_specialization)


@tvm._ffi.register_object
class ILAOps(Object):
    @property
    def ds_tensors(self):
        return self.__getattr__("ds_tensors")

    @property
    def outputs(self):
        return self.__getattr__("outputs")

    @property
    def ra_ila_mapping(self):
        return self.__getattr__("ra_ila_mapping")

    def get_ila(self, key):
        op = key.op if isinstance(key, _tensor.Tensor) else key
        for k, v in self.__getattr__("ra_ila_mapping").items():
            if op == k.op: return v
        raise ValueError('No such key')

# tvm._ffi._init_api("schedule", __name__)
