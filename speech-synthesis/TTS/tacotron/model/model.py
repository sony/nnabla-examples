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
import nnabla.functions as F
from nnabla.initializer import NormalInitializer
import nnabla.parametric_functions as PF

from neu.tts.module import Module

from .ops import Bahdanau_attention
from .ops import encoder_cbhg
from .ops import post_cbhg
from .ops import prenet


class Encoder(Module):
    r"""Encoder implementation."""

    def __init__(self, hparams):
        self._hparams = hparams

    def call(self, inputs):
        r"""
        Args:
            inputs (nn.Variable): An input variable of shape (B, T).

        Returns:
            nn.Variable: Output variable of shape (T, B, C).
        """
        # inputs of shape (B, T)
        hparams = self._hparams
        embedded_inputs = PF.embed(
            inputs, n_inputs=len(hparams.vocab),
            n_features=hparams.symbols_embedding_dim,
            initializer=NormalInitializer(0.3),
            name='embedding'
        )  # (B, T, C)

        prenet_outputs = prenet(
            embedded_inputs,
            layer_sizes=hparams.prenet_channels,
            is_training=self.training, scope='prenet_encoder'
        )  # (B, T, C)

        encoder_outputs = encoder_cbhg(
            F.transpose(prenet_outputs, (0, 2, 1)),
            depth=hparams.encoder_embedding_dim,
            is_training=self.training
        )  # (T, B, C)

        return encoder_outputs


class Decoder(Module):
    r"""RNN Decoder with Attention."""

    def __init__(self, hparams):
        self._hparams = hparams

    def call(self, memory, inputs=None):
        r"""Return mel-spectrogram and attention matrix.

        Args:
            memory(nn.Variable): A 3D tensor of shape (T, B, C).
            inputs(nn.Variable, optional): A 3D tensor with shape of
                [B, T/r, n_mels(*r)]. Shifted log melspectrogram of sound files.
                Defaults to None.

        Returns:
            nn.Variable: The synthetic mel-spectrograms of shape
                (B, Ty/r, r*n_mels).
            nn.Variable: The attention matrix of shape
                (B, Tx, Ty).

        References:
            - https://github.com/Kyubyong/tacotron/
        """
        hp = self._hparams
        bz, mel_shape = hp.batch_size, hp.n_mels * hp.r
        encoder_dim = hp.encoder_embedding_dim

        # initialize input tensor
        input = F.constant(shape=(bz, 1, mel_shape))

        # initialize hidden states
        context = F.constant(shape=(bz, 1, hp.attention_dim))
        hidden = F.constant(shape=(1, 1, bz, encoder_dim))
        h_gru = [F.constant(shape=(1, 1, bz, encoder_dim)),
                 F.constant(shape=(1, 1, bz, encoder_dim))]

        outputs, attends = [], []

        for i in range(hp.n_frames):
            if i > 0:
                input = (
                    outputs[-1] if inputs is None else inputs[:, i-1:i, :])

            # feed a prenet to the input
            input = prenet(input, layer_sizes=hp.prenet_channels,
                           is_training=self.training, scope='prenet_decoder')  # (bz, 1, C)

            # concat the input and context vector
            input = F.concatenate(input, context)  # (bz, 1, 384)

            with nn.parameter_scope('rnn_attention'):
                # calculate the output
                output, hidden = PF.gru(
                    input.reshape((1, bz, -1)), hidden,
                    training=self.training, bidirectional=False
                )  # (1, bz, 256), (1, 1, bz, 256)

            # compute the context and attention vectors
            context, attend = Bahdanau_attention(
                F.transpose(hidden[0], (1, 0, 2)),
                memory, out_features=hp.attention_dim,
                scope='Bahdanau_attention'
            )  # (bz, 1, 256), (bz, 1, T)

            with nn.parameter_scope('rnn_decoder'):
                # concat RNN output and attention context vector
                with nn.parameter_scope('project_to_decoder'):
                    output = F.concatenate(
                        output, F.transpose(context, (1, 0, 2)), axis=2)
                    output = PF.affine(output, encoder_dim,
                                       base_axis=2)  # (1, bz, 256)

                # decoder RNN with residual connection
                for j in range(2):
                    with nn.parameter_scope(f'gru_resisidual_{j}'):
                        out, h_gru[j] = PF.gru(
                            output, h_gru[j], training=self.training, bidirectional=False)
                        output += out  # (1, bz, 256)

                # projector to mels
                with nn.parameter_scope('project_to_mel'):
                    output = F.transpose(output, (1, 0, 2))
                    # (bz, 1, n_mels*r)
                    output = PF.affine(output, mel_shape, base_axis=2)

            outputs.append(output)
            attends.append(attend)

        outputs = F.concatenate(*outputs, axis=1)  # (B, T2, C2)
        attends = F.concatenate(*attends, axis=1)  # (B, T2, T1)

        return outputs, attends


class PostNet(Module):
    r"""The PostNet implementation.

    Args:
        hparams (HPrams): A container containing all hyper-parameters.
    """

    def __init__(self, hparams):
        self._hparams = hparams

    def call(self, inputs):
        r"""
        Args:
            inputs (nn.Variable): An input variable of shape
                (B, T_y/r, n_mels*r)

        Returns:
            nn.Variable: Output variable of shape (T_y, B, n_mels).
        """
        hparams = self._hparams
        inputs = inputs.reshape((inputs.shape[0], -1, hparams.n_mels))
        out = post_cbhg(
            F.transpose(inputs, (0, 2, 1)),
            hparams.n_mels,
            is_training=self.training,
            depth=hparams.symbols_embedding_dim
        )  # (T_y, B, n_mels)

        out = PF.affine(
            F.transpose(out, (1, 0, 2)),
            hparams.n_fft//2+1, base_axis=2, name='affine_post'
        )  # (T_y, B, n_mels)

        return out  # (B, T_y, n_fft/2 + 1)


class Tacotron(Module):
    r"""An implementation of Tacotron.

    Args:
        hparams (HPrams): A container containing all hyper-parameters.

    References:
        [1] Wang et al., 2017. Tacotron: Towards end-to-end speech synthesis.
            arXiv preprint arXiv:1703.10135.
    """

    def __init__(self, hparams):
        self._hparams = hparams
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = PostNet(hparams)

    def call(self, char_inputs, mel_inputs=None):
        r"""Return mel-spectrograms, spectrograms, and attention weights.

        Args:
            char_inputs(nn.Variable): Inputs containing indices of characters.
                This has a shape of(B, Tx).
            mel_inputs(nn.Variable, optional): Inputs containing
                mel-spectrograms of shape(B, Ty/r, n_mels*r).
                Defaults to None. If None, the mel spectrograms are infferred from data.

        Returns:
            nn.Variable: The synthetic mel-spectrograms of shape
                (B, T_y/r, n_mels*r).
            nn.Variable: The synthetic spectrograms of shape
                (B, Ty, n_fft/2 + 1).
            nn.Variable: The attention matrix of shape
                (B, Tx, Ty).
        """
        with nn.parameter_scope('encoder'):
            encoder_outputs = self.encoder(char_inputs)  # (Tx, B, 256)

        with nn.parameter_scope('decoder'):
            mel_outputs, attends = self.decoder(
                F.transpose(encoder_outputs, (1, 0, 2)),
                mel_inputs
            )  # (B, T_y/r, n_mels*r), (B, Ty, Tx)

        with nn.parameter_scope('postprocessing'):
            mag_outputs = self.postnet(mel_outputs)  # (B, T_y, n_fft/2 + 1)

        return mel_outputs, mag_outputs, attends
