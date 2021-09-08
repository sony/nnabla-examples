# Copyright 2020,2021 Sony Corporation.
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

import nnabla.solvers as S
from nnabla.utils.learning_rate_scheduler import BaseLearningRateScheduler


class NoamScheduler(BaseLearningRateScheduler):
    r"""Noam learning rate scheduler.

    Args:
        init_lr (float): Initial learning rate.
        warmup (int): Warmup iteration.
    """

    def __init__(self, init_lr, warmup=4000):
        self.init_lr = init_lr
        self.warmup = warmup

    def get_learning_rate(self, iter):
        r"""Get learning rate with cosine decay based on current iteration.

        Args:
            iter (int): Current iteration (starting with 0).

        Returns:
            float: Learning rate
        """
        step = iter + 1
        return self.init_lr * self.warmup ** 0.5 * min(
            step*self.warmup ** -1.5, step ** -0.5)
