# Copyright 2022 Sony Group Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import contextmanager
from collections import defaultdict
import numpy as np

import nnabla as nn
import nnabla.solvers as S
import nnabla.functions as F
from nnabla.parameter import get_parameter_or_create


@contextmanager
def context_scope(type_config):
    assert type_config in ["float", "half"], \
        "type_config must be either of ['float', 'half']"

    from nnabla.ext_utils import get_extension_context
    current_ctx = nn.get_current_context()
    ctx = get_extension_context(ext_name=current_ctx.backend[0].split(":")[0],
                                device_id=current_ctx.device_id,
                                type_config=type_config)

    nn.set_default_context(ctx)

    yield ctx

    nn.set_default_context(current_ctx)


def force_float(func):
    def wrapped_func(*args, **kwargs):
        with context_scope("float"):
            return func(*args, **kwargs)

    return wrapped_func


def create_ema_parameter(base_name, base_param):
    # create ema parameter whose name is "ema/{base_name}".
    with nn.parameter_scope("ema"):
        ema_param = get_parameter_or_create(base_name,
                                            shape=base_param.shape,
                                            need_grad=False)

    return ema_param


@force_float
def ema_update(ema_param: nn.Variable, train_param: nn.Variable, decay=0.9999):
    assert isinstance(ema_param, nn.Variable)
    assert isinstance(train_param, nn.Variable)

    # apply ema update by inplace operation
    # p_ema = decay * p_ema + (1 - decay) * p_train <- needs copy to assign the result
    #       = p_ema - (1 - decay) * (p_ema - p_train) <- performs the same thing by isub operation

    sub = (1. - decay) * (ema_param.data - train_param.data)
    ema_param.data -= sub


class PackedParameterSolver(object):

    def __init__(self,
                 solver: S.Solver,
                 use_ema: bool):
        self.solver = solver

        # keep original params and states for saving
        self.orig_params = solver.get_parameters()
        self.orig_states = solver.get_states()

        # big params and states
        # Basically we assume params.data is float32, but prams.grad can be either float32 or float16.
        # Therefore, we separately pack params depending on the dtype of grad.
        self._num_pp = 2  # for easy implementation
        self.packed_params = None
        self.packed_states = None
        self.is_packed = False

        # cudnn requires 128 bit and recommends 1024 bit alignment
        # Align size by a multiple of 64 is enough since at least data has a bit size of multiple of 16 bit (float16).
        self.alignment = 64

        # for ema parameters
        self.use_ema = use_ema
        self.packed_ema_params = None

    @staticmethod
    def _get_param_type_based_on_dtype(x: nn.Variable):
        assert x.grad.dtype in (np.float32, np.float16), \
            f"datatype {x.grad.dtype} is not supported."
        return int(x.grad.dtype == np.float32)

    def _align_size(self, size):
        return (size + self.alignment - 1) // self.alignment * self.alignment

    def _create_packed_param_state(self):
        orig_params = self.solver.get_parameters()
        orig_states = self.solver.get_states()

        # 1. compute total size for both parameter and state
        p_total_sizes = [0 for _ in range(self._num_pp)]
        s_total_sizes = [defaultdict(int) for _ in range(self._num_pp)]
        solver_step = None
        for p_key, p in orig_params.items():
            idx = self._get_param_type_based_on_dtype(p)

            # For packed param
            p_total_sizes[idx] += self._align_size(p.size)

            # For packed states
            state = orig_states[p_key]
            if solver_step is None:
                solver_step = state.t
            else:
                # All solver steps must be identical
                assert solver_step == state.t

            # Each param could have more than one corresponding states.
            for s_name, pstate in state.pstate.items():
                s_total_sizes[idx][s_name] += self._align_size(pstate.size)

        # 2. create large variable for packing
        # parameter
        self.packed_params = [nn.Variable(shape=(x, ))
                              for x in p_total_sizes]

        if self.use_ema:
            self.packed_ema_params = [nn.Variable(shape=(x, ))
                                      for x in p_total_sizes]

        # state
        packed_pstates = [dict() for _ in range(self._num_pp)]
        for i, sname_to_size_dict in enumerate(s_total_sizes):
            for sname, size in sname_to_size_dict.items():
                packed_pstates[i][sname] = nn.Variable(shape=(size, ))

        # 3. assign a segment to each param/state.
        p_cur_sizes = [0 for _ in range(self._num_pp)]
        s_cur_sizes = [defaultdict(int) for _ in range(self._num_pp)]

        for p_key, p in orig_params.items():
            # get packed parameter id (0: grad is float16, 1: grad is float32)
            idx = self._get_param_type_based_on_dtype(p)

            # 3.1 replace parameter
            # select a packed param for the current one
            packed_param = self.packed_params[idx]
            p_cur_size = p_cur_sizes[idx]

            # get narrowed array for the current parameter
            data_narrowed = packed_param.data.narrow(
                0, p_cur_size, p.size).view(p.shape)
            grad_narrowed = packed_param.grad.narrow(
                0, p_cur_size, p.size).view(p.shape)

            # copy data and grad from the original parameter
            if p.data.dtype != "float32":
                raise ValueError(f"p.data.dtype != float32 for {p_key}.")
            # set use_current_context as False to keep the dtype of the original parameter.
            data_narrowed.copy_from(p.data, use_current_context=False)
            grad_narrowed.copy_from(p.grad, use_current_context=False)

            # replace ndarray
            p.data = data_narrowed
            p.grad = grad_narrowed

            # ema
            if self.use_ema:
                # create (or get) ema param
                p_ema = create_ema_parameter(p_key, p)

                # replace ndarray
                packed_ema_param = self.packed_ema_params[idx]
                ema_data_narrowed = packed_ema_param.data.narrow(
                    0, p_cur_size, p.size).view(p.shape)

                ema_data_narrowed.copy_from(
                    p_ema.data, use_current_context=False)
                p_ema.data = ema_data_narrowed

            # update total size
            p_cur_sizes[idx] += self._align_size(p.size)

            # 3.2 replace state
            # get states for the current param
            state = orig_states[p_key]

            # replace ndarray for each state
            for s_name, pstate in state.pstate.items():
                s_cur_size = s_cur_sizes[idx][s_name]
                pstate_narrowed = packed_pstates[idx][s_name].data.narrow(0,
                                                                          s_cur_size,
                                                                          pstate.size)
                pstate_narrowed = pstate_narrowed.view(pstate.shape)

                # (20221103) Always use float context since there is no way to check whether pstate has array or not.
                with context_scope("float"):
                    pstate_narrowed.copy_from(pstate.data)
                pstate.data = pstate_narrowed

                s_cur_sizes[idx][s_name] += self._align_size(pstate.size)

        # create SolverState object from packed states
        self.packed_states = []
        for idx in range(self._num_pp):
            solver_state = S.SolverState()
            solver_state.t = solver_step
            solver_state.pstate = {}

            for s_name, pstate in packed_pstates[idx].items():
                solver_state.pstate[s_name] = pstate

            self.packed_states.append(solver_state)

    def _packing(self):
        if self.is_packed:
            return

        # create packed param and state
        self._create_packed_param_state()

        # keep original parameters and states for saving
        self.orig_params = self.solver.get_parameters()
        self.orig_states = self.solver.get_states()

        # reset solver to have new packed parameters
        self.solver.clear_parameters()
        packed_params_dict = {}
        packed_states_dict = {}
        for idx in range(self._num_pp):
            p = self.packed_params[idx]
            s = self.packed_states[idx]

            # skip if empty, otherwise solver method raises due to empty array.
            if p.size == 0:
                assert len(s.pstate) == 0
                continue

            packed_params_dict[f"PackedParam{idx}"] = p
            packed_states_dict[f"PackedParam{idx}"] = s

        self.solver.set_parameters(packed_params_dict)
        self.solver.set_states(packed_states_dict)

        self.is_packed = True

    def updata_ema_params(self, decay=0.9999):
        if not self.use_ema:
            return

        if not self.is_packed:
            self._packing()

        for p_ema, p_train in zip(self.packed_ema_params, self.packed_params):
            if p_ema.size == 0:
                assert p_train.size == 0
                continue

            # make sure to keep data
            p_ema.persistent = True
            p_train.persistent = True

            ema_update(p_ema, p_train, decay=decay)

    def set_parameters(self, *args, **kwargs):
        raise NotImplementedError(
            "All parameters must be set to solver object before passing solver to PackedParameterSolver.")

    def set_learning_rate(self, lr):
        self.solver.set_learning_rate(lr)

    def zero_grad(self):
        self.solver.zero_grad()

    def check_inf_or_nan_grad(self):
        self._packing()
        return self.solver.check_inf_or_nan_grad()

    def update(self, *args, **kwargs):
        self._packing()
        self.solver.update(*args, **kwargs)

    def scale_grad(self, *args, **kwargs):
        self._packing()
        self.solver.scale_grad(*args, **kwargs)

    def clip_by_norm(self, *args, **kwargs):
        self._packing()
        self.solver.clip_by_norm(*args, **kwargs)

    # Functions which need to convert the packed param to the original params
    def save_states(self, path):
        # keep packed states
        packed_states = self.solver.get_states()
        packed_params = self.solver.get_parameters()

        # reset original states to solver
        self.solver.set_parameters(self.orig_params, reset=True)

        # set current timestep for all state
        packed_t = None
        for idx in range(self._num_pp):
            s = self.packed_states[idx]
            if packed_t is None:
                packed_t = s.t
            else:
                assert packed_t == s.t

        for k, s in self.orig_states.items():
            s.t = packed_t

        self.solver.set_states(self.orig_states)
        self.solver.save_states(path)

        # revert packed params
        self.solver.set_parameters(packed_params, reset=True)
        self.solver.set_states(packed_states)

    def get_parameters(self):
        return self.orig_params

    def get_states(self):
        # set current timestep for all state
        packed_t = None
        for idx in range(self._num_pp):
            s = self.packed_states[idx]
            if packed_t is None:
                packed_t = s.t
            else:
                assert packed_t == s.t

        for k, s in self.orig_states.items():
            if s.t == packed_t:
                # no need to restore timestep
                break
            s.t = packed_t

        return self.orig_states

    def load_states(self, path):
        if not self.is_packed:
            # solver must have origianl parameters and states.
            # call solver.load_stats directely
            self.solver.load_states(path)
            return

        # otherwise, replace states and re-packing again

        # reset original states to solver
        self.solver.set_parameters(self.orig_params, reset=True)
        self.solver.load_states(path)

        # re-pack parameters
        self.is_packed = False
        self._packing()
