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
import nnabla.communicators as C
import nnabla.functions as F
import numpy as np
from nnabla.ext_utils import get_extension_context
from scipy.interpolate import interp1d


def create_float_context(ctx):
    ctx_float = get_extension_context(ctx.backend[0].split(':')[
                                      0], device_id=ctx.device_id)
    return ctx_float


def model_numel(model, grad_only=True):
    r"""Returns the number of parameters in model.

    Args:
        model (nn.Module): Input model.
        grad_only (bool): Whether to consider only `with_grad` parameters.

    Returns:
        int: Number of parameters in model.
    """
    ret = 0
    for p in model.get_parameters(grad_only=grad_only).values():
        ret += np.prod(p.shape)

    return ret


def get_alignment(tier, sr, hop_length):
    r"""Returns alignment.

    Args:
        tier ([type]): [description]
        sr ([type]): [description]
        hop_length ([type]): [description]

    Returns:
        List[str]: List of phonemes.
        List[int]: List of durations.
    """
    phones, durations = list(), list()
    for t in tier._objects:
        s, e, p = t.start_time, t.end_time, t.text
        d = int(np.round(e * sr / hop_length) - np.round(s * sr / hop_length))
        phones.append(p)
        durations.append(d)

    return phones, durations


def preprocess_pitch(pitch, duration):
    r"""Preprocess pitch.

    Args:
        pitch (np.ndarray): Pitch contour extracted from pyworld.
        duration (List[int]): List of durations.

    Returns:
        np.ndarray: Preprocessed pitch.
    """
    pitch = pitch[:sum(duration)]

    # interpolate unvoiced frames
    valids = np.where(pitch != 0)[0]
    interp_fn = interp1d(
        valids,
        pitch[valids],
        fill_value=(pitch[valids[0]], pitch[valids[-1]]),
        bounds_error=False,
    )
    pitch = interp_fn(np.arange(0, len(pitch)))

    # TODO: log transform

    # compute pitch per phoneme
    pitch_per_phone, start = list(), 0
    for d in duration:
        v = np.mean(pitch[start:start + d]) if d > 0 else 0
        pitch_per_phone.append(v)
        start += d
    pitch = np.array(pitch_per_phone)

    return pitch


def preprocess_energy(energy, duration):
    """Preprocess energy.

    Args:
        energy (np.ndarray)): Input energy.
        duration (List[int]): List of durations.

    Returns:
        np.ndarray: Preprocessed energy.
    """
    energy = energy[:sum(duration)]

    # compute energy per phoneme
    energy_per_phone, start = list(), 0
    for d in duration:
        v = np.mean(energy[start:start + d]) if d > 0 else 0
        energy_per_phone.append(v)
        start += d
    energy = np.array(energy_per_phone)

    return energy


def get_mask(seq_len, max_len):
    """Gets a mask with 1 filled over `out-of-range` position.

    Args:
        seq_len (nn.Variable): Length variable of shape (B, 1)
        max_len (int): Maximum length.

    Returns:
        nn.Variable: Mask variable of shape (B, max_len, 1).
    """
    mask = F.arange(0, max_len).reshape((1, -1))
    mask = F.greater_equal(mask, seq_len)
    mask = F.reshape(mask, shape=mask.shape + (1,))

    return mask
