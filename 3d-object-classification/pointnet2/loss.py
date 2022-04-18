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

import nnabla as nn
import nnabla.functions as F


def classification_loss(pred_logit: nn.Variable, label: nn.Variable) -> nn.Variable:
    """classification loss

    Args:
        pred_logit (nn.Variable): pred logit, shape(batch, num_classes)
        label (nn.Variable): label, shape(batch, 1)
        transformation_mat (nn.Variable): label, shape(batch, K, K)

    Returns:
        nn.Variable: loss
    """
    cross_entropy_loss = F.softmax_cross_entropy(pred_logit, label)
    classify_loss = F.mean(cross_entropy_loss)
    return classify_loss
