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

import nnabla as nn
import numpy as np

from ray_tracer import bisection, secant

import pytest


def f(x, a=1, alpha=0, beta=5, gamma=10):
    return a * (x - alpha) * (x - beta) * (x - gamma)


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("method", [bisection, secant])
@pytest.mark.parametrize("max_post_itr", [10])
@pytest.mark.parametrize("a, alpha, beta, gamma, eps0, eps1",
                         [(1, 0, 5, 10, 1, 2)])
@pytest.mark.parametrize("same", [False, True])
def test_secant(seed, method, max_post_itr,
                a, alpha, beta, gamma, eps0, eps1, same):

    with nn.auto_forward():
        start = alpha + eps0
        finish = gamma - eps1 if not same else start
        x0 = nn.Variable.from_numpy_array(start).data
        x1 = nn.Variable.from_numpy_array(finish).data
        x0, x1 = method(x0, x1, f, max_post_itr)

    expected = beta if not same else start
    np.testing.assert_allclose(x0.data, expected, atol=1e-2)
