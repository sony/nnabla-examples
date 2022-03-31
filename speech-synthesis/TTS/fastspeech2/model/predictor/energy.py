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

from pathlib import Path

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import numpy as np

from utils.ops import bucketize

from neu.tts.module import Module
from .ops import predict


class EnergyPredictor(Module):
    r"""Energy Predictor module.

    Args:
        hp (HParams): Hyper-parameters.
    """

    def __init__(self, hp):
        self.hp = hp
        stat = np.load(Path(hp.precomputed_path) / 'statistics.npz')
        p_min = (stat['min_energy'] - stat['mean_energy']) / stat['std_energy']
        p_max = (stat['max_energy'] - stat['mean_energy']) / stat['std_energy']
        # define energy bins
        bins = nn.Variable.from_numpy_array(
            np.linspace(p_min, p_max, hp.n_bins - 1), need_grad=False)
        self.energy_bins = F.tile(bins, (hp.batch_size, 1))

    def call(self, x, mask=None, target=None, control=1.0):
        r"""Returns energy prediction.

        Args:
            x (nn.Variable): Input variable of shape (B, max_len, dim).
            mask (nn.Variable, optional): Mask variable of shape
                (B, max_len, 1). Defaults to None.
            target (nn.Variable, optional): Target energy variable of shape
                (B, max_len). Defaults to None.
            control (float, optional): Scale controling energy.
                Defaults to 1.0

        Returns:
            nn.Variable: Energy prediction variable of shape (B, max_len).
            nn.Variable: Energy embedding variable of shape (B, max_len, dim).
        """
        hp = self.hp
        out = predict(x, mask, self.hp, self.training)
        if target is None:  # inference
            target = out * control

        idx = bucketize(target, self.energy_bins)
        with nn.parameter_scope("embedding"):
            embedings = PF.embed(idx, hp.n_bins, n_features=hp.encoder_hidden)

        return out, embedings
