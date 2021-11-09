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

from typing import Dict, Tuple
import numpy as np

import nnabla as nn
import nnabla.functions as F


def classification_loss_with_orthogonal_loss(
    pred_logit: nn.Variable, label: nn.Variable, transformation_mat: nn.Variable, reg_weight=0.001
) -> Tuple[nn.Variable, Dict[str, nn.Variable]]:
    """classification loss with orthogonal loss

    Args:
        pred_logit (nn.Variable): pred logit, shape(batch, num_classes)
        label (nn.Variable): label, shape(batch, 1)
        transformation_mat (nn.Variable): label, shape(batch, K, K)

    Returns:
        Tuple[nn.Variable, Dict[str, nn.Variable]]: loss and internal loss
    """
    cross_entropy_loss = F.softmax_cross_entropy(pred_logit, label)
    classify_loss = F.mean(cross_entropy_loss)

    # Enforce the transformation as orthogonal matrix
    mat_squared = F.batch_matmul(
        transformation_mat, F.transpose(transformation_mat, (0, 2, 1)))
    batch_size, k, _ = transformation_mat.shape
    target_array = np.tile(np.eye(k, dtype=np.float32), (batch_size, 1, 1))
    target = nn.Variable.from_numpy_array(target_array)
    mat_diff = mat_squared - target

    # Frobenius norm
    mat_diff = F.reshape(mat_diff, (batch_size, -1))
    mat_loss = F.mean(F.norm(mat_diff, axis=1))

    return classify_loss + mat_loss * reg_weight, {
        "classify_loss": classify_loss,
        "mat_loss": mat_loss,
        "mat_diff": mat_diff,
    }
