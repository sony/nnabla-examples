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

from neu.tts.module import Module
from .ops import predict


class DurationPredictor(Module):
    r"""Duration Predictor module.

    Args:
        hp (HParams): Hyper-parameters.
    """

    def __init__(self, hp):
        self.hp = hp

    def call(self, x, mask):
        r"""Returns duration prediction.

        Args:
            x (nn.Variable): Input variable of shape (B, max_len, dim).
            mask (nn.Variable, optional): Mask variable of shape
                (B, max_len, 1). Defaults to None.
        Returns:
            nn.Variable: Duration prediction variable of shape (B, max_len).
                Durations are measured in log scale.
        """
        x = predict(x, mask, self.hp, self.training)
        return x
