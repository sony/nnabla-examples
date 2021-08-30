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
import nnabla.solvers as S


def train_model(model, data, labels):
    model(data)
    solver = S.Adam(alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-08)
    solver.set_parameters(nn.get_parameters())
    for i in range(1500):
        out = model(data)
        loss = F.categorical_cross_entropy(out, labels)
        loss.forward()
        solver.zero_grad()
        loss.backward()
        solver.update()
