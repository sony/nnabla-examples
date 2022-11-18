# Copyright 2021 Sony Group Corporation.
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
from typing import Union, List
from collections import OrderedDict

import nnabla as nn
import nnabla.functions as F
import numpy as np
from nnabla.parameter import get_parameter_or_create

from config import TrainConfig

# Shape handler


class Shape4D(object):
    def __init__(self, shape: Union[tuple, list], channel_last: bool):

        assert isinstance(shape, (tuple, list)) and len(shape) == 4, \
             f"shape must be a tuple having 4 elements, but {shape} is given."

        c_axis = 3 if channel_last else 1
        h_axis = 1 if channel_last else 2
        w_axis = h_axis + 1

        self.b = shape[0]
        self.c = shape[c_axis]
        self.h = shape[h_axis]
        self.w = shape[w_axis]

    def __str__(self) -> str:
        return f"(b={self.b}, c={self.c}, h={self.h}, w={self.w})"

    def get_as_tuple(self, subscripts: str) -> tuple:
        assert isinstance(subscripts, str)

        ret = []
        for s in subscripts:
            assert s.lower(
            ) in "bchw", f"Unknown axis `{s}` is specified. Subscripts must be consist of [`b`,`c`,`h`,`w`]."
            ret.append(getattr(self, s.lower()))

        return ret[0] if len(ret) == 1 else tuple(ret)

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, (tuple, list)):
            if len(__o) != 4:
                return False

            __o = Shape4D(__o)
            return self == __o

        if isinstance(__o, Shape4D):
            return self.b == __o.b and self.c == __o.c and \
                self.h == __o.h and self.w == __o.w

        return False


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


def get_lr_scheduler(conf: TrainConfig):
    if conf.lr_scheduler is None:
        return None
    elif conf.lr_scheduler == "cosine":
        from neu.learning_rate_scheduler import EpochCosineLearningRateScheduler
        return EpochCosineLearningRateScheduler(base_lr=conf.lr,
                                                epochs=conf.n_iters,
                                                warmup_epochs=10000)

    raise ValueError(f"scheduler name '{conf.lr_scheduler}' is not supported.")


def sum_grad_norm(params):
    norm = nn.NdArray()
    norm.zero()

    for p in params:
        assert isinstance(p, nn.Variable) and not p.grad.clear_called
        norm += F.sum(p.grad ** 2)

    return np.sqrt(norm.data)


def create_ema_op(params, ema_decay=0.9999):
    """
    Define exponential moving average update for trainable params.
    """
    # Have to use float for ema update.
    # Otherwise ema update doesn't work nicely.
    with context_scope("float") as float_ctx:
        def ema_update(p_ema, p_train):
            return F.assign(p_ema, ema_decay * p_ema + (1. - ema_decay) * p_train)

        ops = []
        ema_params = OrderedDict()
        with nn.parameter_scope("ema"):
            for name, p_train in params.items():
                p_ema = get_parameter_or_create(
                    name, shape=p_train.shape, need_grad=False)
                p_ema.data.copy_from(p_train.data)  # initialize
                p_ema.data.cast(float, float_ctx)
                
                # construct ema_op
                ops.append(ema_update(p_ema, p_train))
                
                ema_params["/".join(["ema", name])] = p_ema

        return F.sink(*ops), ema_params


def to_cpu(vars: List[nn.Variable]):
    from nnabla.ext_utils import get_extension_context
    cpu_ctx = get_extension_context(
        ext_name="cpu",
        type_config="float"
    )

    for var in vars:
        var.data.cast(float, cpu_ctx)


# neu extention
# todo: move this to neu

def init_checkpoint_queue(path):
    import re
    import os
    all_files = os.listdir(path)

    # get saved iters in ascending order.
    # checkpoint files should be named as "checkpoint_{iter}.json".
    checkpoints = [x for x in all_files if x.startswith("checkpoint")]
    saved_iters = sorted(
        [int(re.findall("checkpoint_(\d+).json", x)[0]) for x in checkpoints])

    # trace all saved files
    from collections import defaultdict
    saved_paths_per_iter = defaultdict(list)
    for filename in all_files:
        if not filename.endswith(".h5"):
            continue

        # filename should be "{identifier}_{iter}.h5".
        filename_wo_ext = os.path.splitext(filename)[0]
        iter = int(filename_wo_ext.split("_")[-1])

        saved_paths_per_iter[iter].append(os.path.join(path, filename))

    from neu.checkpoint_util import prev_save_paths
    for iter in saved_iters:
        saved_paths = saved_paths_per_iter[iter]

        cp_path = os.path.join(path, f"checkpoint_{iter}.json")
        assert os.path.exists(cp_path), f"{cp_path} doesn't exist."
        saved_paths.append(cp_path)

        prev_save_paths.put(saved_paths)
