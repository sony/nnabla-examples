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
import nnabla.parametric_functions as PF

from utils.ops import masked_fill
from utils.text import wdict

from neu.tts.module import Module
from .transformer.layer import FFTBlock
from .transformer.ops import get_positional_encoding


class Encoder(Module):
    r"""Encoder module.

    Args:
        hp (Hparams): Hyper-parameters.
    """

    def __init__(self, hp):
        self.hp = hp
        self.pos_enc = get_positional_encoding(
            hp.max_len_phone, hp.encoder_hidden
        )
        for i in range(hp.encoder_layer):
            setattr(
                self, f"fft_block_{i}",
                FFTBlock(
                    n_head=hp.encoder_head,
                    n_hidden=hp.encoder_hidden,
                    conv_filter_size=hp.conv_filter_size,
                    conv_kernel_size=hp.conv_kernel_size,
                    dropout=hp.encoder_dropout
                )
            )

    def call(self, x, mask):
        r"""Compute embeddings from phonemes.

        Args:
            x (nn.Variable): Input variable of phonemes (B, max_len).
            mask (nn.Variable): Mask vairable of shape (B, max_len, 1).

        Returns:
            nn.Variable: Output variable of shape (B, max_len, dim).
        """
        hp = self.hp
        with nn.parameter_scope("embedding"):
            x = PF.embed(
                x, len(wdict.___VALID_SYMBOLS___),
                n_features=hp.encoder_hidden)  # check `padding_idx`
            x = x + F.tile(self.pos_enc, (hp.batch_size, 1, 1))

        if mask is not None:
            mask = F.transpose(mask, (0, 2, 1))

        for i in range(hp.encoder_layer):
            x = getattr(self, f"fft_block_{i}")(x, mask)

        # remove masked output
        if mask is not None:
            x = masked_fill(x, F.transpose(mask, (0, 2, 1)))

        return x
