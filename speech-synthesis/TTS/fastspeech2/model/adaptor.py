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

from utils.ops import regulate

from neu.tts.module import Module
from .predictor.duration import DurationPredictor
from .predictor.energy import EnergyPredictor
from .predictor.pitch import PitchPredictor


class Adaptor(Module):
    r"""Adaptor module.

    Args:
        hp (Hparams): Hyper-parameters.
    """

    def __init__(self, hp):
        self.hp = hp
        self.energy = EnergyPredictor(hp)
        self.pitch = PitchPredictor(hp)
        self.duration = DurationPredictor(hp)

    def call(self, x, mask_phone, target_pitch=None, target_energy=None,
             target_duration=None, control_pitch=1.0, control_energy=1.0,
             control_duration=1.0):
        """Returns variance adaptor.

        Args:
            x (nn.Variable): Input variable of shape (B, max_len, dim).
            mask_phone (nn.Variable): Mask variable of shape
                (B, max_len, 1). Defaults to None.
            target_pitch (nn.Variable, optional): Target pitch variable of
                shape (B, max_len). Defaults to None.
            target_energy (nn.Variable, optional): Target energy variable of
                shape (B, max_len). Defaults to None.
            target_duration (nn.Variable, optional): Target duration variable
                of shape (B, max_len). Defaults to None.
            control_pitch (float, optional): Scale controling pitch.
                Defaults to 1.0.
            control_energy (float, optional): Scale controling energy.
                Defaults to 1.0.
            control_duration (float, optional): Scale controling duration.
                Defaults to 1.0.

        Returns:
            nn.Variable: Output variable of shape (B, max_mel, dim).
            nn.Variable: Log duration of shape (B, max_len).
            nn.Variable: Pitch prediction of shape (B, max_len).
            nn.Variable: Evergy prediction of shape (B, max_len).
            nn.Variable: Target duration of shape (B, max_len)
        """
        hp = self.hp
        skip = x
        with nn.parameter_scope("duration_predictor"):
            log_duration = self.duration(skip, mask_phone)
            if target_duration is None:
                duration = (F.exp(log_duration) - 1.0) * control_duration
                target_duration = F.relu(F.round(duration))

        with nn.parameter_scope("pitch_predictor"):
            pred_pitch, embed_pitch = self.pitch(
                skip, mask_phone, target_pitch, control_pitch
            )
            x = x + embed_pitch

        with nn.parameter_scope("energy_predictor"):
            pred_energy, embed_energy = self.energy(
                skip, mask_phone, target_energy, control_energy
            )
            x = x + embed_energy

        with nn.parameter_scope("length_regulator"):
            x = regulate(x, target_duration, hp.max_len_mel)

        return x, log_duration, pred_pitch, pred_energy, target_duration
