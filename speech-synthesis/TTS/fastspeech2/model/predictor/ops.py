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
import nnabla.parametric_functions as PF

from utils.ops import masked_fill


def predict(x, mask, hp, training):
    r"""Predicts an output for each frame in input.

    Args:
        x (nn.Variable): Input variable of shape (B, max_len, dim).
        mask (nn.Variable): Mask variable of shape (B, mask_len, 1).

    Returns:
        nn.Variable: Output variable of shape (B, max_len).
    """
    fz, kz = hp.predictor_filter_size, hp.predictor_kernel_size

    x = F.transpose(x, (0, 2, 1))
    for i in range(hp.predictor_blocks):
        with nn.parameter_scope(f"block_{i}"):
            x = PF.convolution(x, fz, (kz,), ((kz-1)//2,), name='conv')
            x = F.relu(x)
            x = PF.layer_normalization(x, batch_axis=(0, 2), name='norm')
            if training:
                x = F.dropout(x, hp.predictor_dropout)

    x = F.transpose(x, (0, 2, 1))
    with nn.parameter_scope("projection"):
        x = PF.affine(x, 1, base_axis=2)

    # set masked values to 0
    x = masked_fill(x, mask, 0)
    x = F.reshape(x, x.shape[:2])

    return x
