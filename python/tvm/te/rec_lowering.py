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

def lower_dyn_batch(ops, rec_vars, leaf_specialization, is_list = False, homogenous_batch = False, batch_size = -1, length = -1):
    return _ffi_api.LowerDynamicBatching(ops, rec_vars.num_nodes_var,
                                         rec_vars.num_batches_var,
                                         rec_vars.max_batch_len_var,
                                         rec_vars.max_child_num_var,
                                         rec_vars.max_int_idx_var,
                                         leaf_specialization, is_list,
                                         homogenous_batch, batch_size, length)

class StaticRecVars:
    def __init__(self):
        self.num_nodes_var = tvm.tir.Var('num_nodes', 'int32')
        self.max_tree_len_var = tvm.tir.Var('max_tree_len', 'int32')
        self.max_child_num_var = tvm.tir.Var('max_child_num', 'int32')

def lower_static_batch(ops, num_trees, rec_vars):
    if isinstance(num_trees, int): num_trees = tvm.tir.IntImm("int32", num_trees)
    return _ffi_api.LowerStaticBatching(ops, rec_vars.num_nodes_var, num_trees,
                                        rec_vars.max_tree_len_var, rec_vars.max_child_num_var)

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
        if not isinstance(key, _tensor.Tensor):
            raise ValueError('Need a tensor')
        op = key.op
        for k, v in self.__getattr__("ra_ila_mapping").items():
            if op == k.op and key.value_index == k.value_index: return v
        raise ValueError('No such key')

    def get_ds_dim(self, name):
        for k, v in self.__getattr__("ds_dimensions").items():
            if name == k: return v
        raise ValueError('No such dim')

# tvm._ffi._init_api("schedule", __name__)
