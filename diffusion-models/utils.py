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


import nnabla as nn
import nnabla.functions as F
import numpy as np
from nnabla.parameter import get_parameter_or_create


def get_warmup_lr(max_lr, warmup_epoch, step):
    # linear warmup
    if warmup_epoch == 0 or step >= warmup_epoch:
        return max_lr

    return max_lr * step / warmup_epoch


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
    def ema_update(p_ema, p_train):
        return F.assign(p_ema, ema_decay * p_ema + (1. - ema_decay) * p_train)

    ops = []
    with nn.parameter_scope("ema"):
        for name, p_train in params.items():
            p_ema = get_parameter_or_create(
                name, shape=p_train.shape, need_grad=False)
            p_ema.data.copy_from(
                p_train.data, use_current_context=False)  # initialize
            ops.append(ema_update(p_ema, p_train))

        ema_params = nn.get_parameters(grad_only=False)

    return F.sink(*ops), ema_params
