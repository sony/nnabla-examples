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
import numpy as np


def get_positional_encoding(n_pos, dim):
    r"""Compute positional encodings

    Args:
        n_pos (int): Length of sequence.
        dim (int): Dimension of embeddings.

    Returns:
        nn.Variable: Output variable.
    """
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / dim)
         for j in range(dim)] for pos in range(n_pos)])
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])

    return nn.Variable.from_numpy_array(position_enc, need_grad=False)
