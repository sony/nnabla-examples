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

from neu.tts.module import Module
import nnabla as nn
import nnabla.functions as F
from nnabla.initializer import UniformInitializer
import nnabla.parametric_functions as PF
import numpy as np

from .ops import affine_norm
from .ops import conv_norm
from .ops import location_sensitive_attention
from .ops import prenet


class Encoder(Module):
    r"""Encoder implementation.

    Args:
        hparams (HPrams): A container containing all hyper-parameters.
    """

    def __init__(self, hparams):
        self._hparams = hparams

    def call(self, inputs):
        r"""Encoder layer.
        Args:
            inputs (nn.Variable): An input variable of shape (B, T) indicates indices
                of character embeddings.

        Returns:
            nn.Variable: Output variable of shape (T, B, C).
        """
        hp = self._hparams
        with nn.parameter_scope('embeddings'):
            val = np.sqrt(6.0 / (len(hp.vocab) + hp.symbols_embedding_dim))
            inputs = PF.embed(
                inputs, n_inputs=len(hp.vocab), n_features=hp.symbols_embedding_dim,
                initializer=UniformInitializer(lim=(-val, val))
            )  # (B, T, C=512)

        with nn.parameter_scope('ngrams'):
            out = inputs
            for i in range(hp.encoder_n_convolutions):
                with nn.parameter_scope(f'filter_{i}'):
                    out = conv_norm(
                        out,
                        out_channels=hp.encoder_embedding_dim,
                        kernel_size=hp.encoder_kernel_size,
                        padding=(hp.encoder_kernel_size - 1) // 2,
                        bias=False, stride=1, dilation=1,
                        w_init_gain='relu', scope='conv_norm', channel_last=True
                    )  # (B, C=512, T)
                    out = PF.batch_normalization(
                        out, batch_stat=self.training, axes=[2])
                    out = F.relu(out)
                    if self.training:
                        # (B, C=512, T) --> (B, T, C=512)
                        out = F.dropout(out, 0.5)

        with nn.parameter_scope('lstm_encoder'):
            out = F.transpose(out, (1, 0, 2))  # (2, 0, 1))
            h = F.constant(shape=(2, 2, hp.batch_size,
                                  hp.encoder_embedding_dim//2))
            c = F.constant(shape=(2, 2, hp.batch_size,
                                  hp.encoder_embedding_dim//2))
            out, _, _ = PF.lstm(
                out, h, c, training=self.training, bidirectional=True)

        return out  # (T, B, C=512)


class Decoder(Module):
    r"""RNN Decoder with Attention.
    Args:
        hparams (HPrams): A container containing all hyper-parameters.
    """

    def __init__(self, hparams):
        self._hparams = hparams

    def call(self, memory, decoder_inputs=None):
        r"""Return mel-spectrograms, gate outputs and an attention matrix.

        Args:
            memory (nn.Variable): A 3D tensor of shape (B, T, C).
            decoder_inputs (nn.Variable, optional): A 3D tensor with shape of (B, T/r, r*n_mels).
                Shifted log melspectrogram of sound files. Defaults to None.

        Returns:
            nn.Variable: The synthetic mel-spectrograms of shape (B, Ty/r, r*n_mels).
            nn.Variable: The gate outputs of shape (B, Ty).
            nn.Variable: The attention matrix of shape (B, Tx, Ty).
        """
        hp = self._hparams
        mel_shape = hp.n_mels * hp.r

        # initialize decoder states
        decoder_input = F.constant(shape=(hp.batch_size, 1, mel_shape))
        decoder_hidden = F.constant(
            shape=(1, 1, hp.batch_size, hp.decoder_rnn_dim))
        decoder_cell = F.constant(
            shape=(1, 1, hp.batch_size, hp.decoder_rnn_dim))

        # initialize attention states
        attention_weights = F.constant(shape=(hp.batch_size, 1, hp.text_len))
        attention_weights_cum = F.constant(
            shape=(hp.batch_size, 1, hp.text_len))
        attention_context = F.constant(
            shape=(hp.batch_size, 1, hp.encoder_embedding_dim))
        attention_hidden = F.constant(
            shape=(1, 1, hp.batch_size, hp.attention_rnn_dim))
        attention_cell = F.constant(
            shape=(1, 1, hp.batch_size, hp.attention_rnn_dim))

        # store outputs
        mel_outputs, gate_outputs, alignments = [], [], []

        for i in range(hp.mel_len):
            if i > 0:
                decoder_input = (
                    mel_outputs[-1] if decoder_inputs is None else decoder_inputs[:, i-1:i, :])
                if decoder_inputs is None:
                    decoder_input = decoder_input[None, ...]
            # decoder of shape (B, 1, prenet_channels=256)
            decoder_input = prenet(
                decoder_input, hp.prenet_channels, is_training=self.training, scope='prenet')

            with nn.parameter_scope('attention_rnn'):
                # cell_input of shape (B, 1, prenet_channels[-1] + C=768)
                cell_input = F.concatenate(
                    decoder_input, attention_context, axis=2)
                _, attention_hidden, attention_cell = PF.lstm(
                    F.transpose(cell_input, (1, 0, 2)),
                    attention_hidden, attention_cell,
                    training=self.training, name='lstm_attention'
                )  # (1, 1, B, attention_hidden), (1, 1, B, attention_hidden)
                if self.training:
                    attention_hidden = F.dropout(
                        attention_hidden, hp.p_attention_dropout)

            with nn.parameter_scope('location_attention'):
                attention_weights_cat = F.concatenate(
                    attention_weights, attention_weights_cum, axis=1)
                attention_context, attention_weights = location_sensitive_attention(
                    F.transpose(attention_hidden[0], (1, 0, 2)),
                    memory, attention_weights_cat,
                    attention_location_kernel_size=hp.attention_location_kernel_size,
                    attention_n_filters=hp.attention_location_n_filters,
                    attention_dim=hp.attention_dim,
                    is_training=self.training, scope='ls_attention'
                )
                attention_weights_cum += attention_weights
                alignments.append(attention_weights)

            with nn.parameter_scope('decoder_rnn'):
                # (1, B, attention_rnn_dim + encoder_embedding_dim)
                inp_decoder = F.concatenate(attention_hidden[0], F.transpose(
                    attention_context, (1, 0, 2)), axis=2)
                _, decoder_hidden, decoder_cell = PF.lstm(
                    inp_decoder, decoder_hidden, decoder_cell,
                    training=self.training, name='lstm_decoder'
                )
                if self.training:
                    decoder_hidden = F.dropout(
                        decoder_hidden, hp.p_decoder_dropout)

            with nn.parameter_scope('projection'):
                proj_input = F.concatenate(
                    decoder_hidden[0, 0],
                    F.reshape(attention_context,
                              (hp.batch_size, -1), inplace=False),
                    axis=1
                )  # (B, decoder_rnn_dim + encoder_embedding_dim)
                decoder_output = affine_norm(proj_input, mel_shape, base_axis=1, with_bias=True,
                                             w_init_gain='affine', scope='affine')
                mel_outputs.append(decoder_output)

            with nn.parameter_scope('gate_prediction'):
                gate_prediction = affine_norm(proj_input, 1, base_axis=1, with_bias=True,
                                              w_init_gain='sigmoid', scope='affine')
                gate_outputs.append(gate_prediction)

        # (B, T2, n_mels*r)
        mel_outputs = F.stack(*mel_outputs, axis=1)
        gate_outputs = F.concatenate(*gate_outputs, axis=1)   # (B, T2)
        alignments = F.concatenate(*alignments, axis=1)       # (B, T1, T2)

        return mel_outputs, gate_outputs, alignments


class PostNet(Module):
    r"""The PostNet implementation.

    Five 1D-convolution with 512 channels and kernel size 5.

    Args:
        hparams (HPrams): A container containing all hyper-parameters.
    """

    def __init__(self, hparams):
        self._hparams = hparams

    def call(self, inputs):
        """Return a mel-spectrogram.

        Args:
            inputs (nn.Variable): A mel-spectrogram of shape (B, T/r, n_mels*r).

        Returns:
            nn.Variable: The resulting spectrogram of shape (B, T/r, n_mels*r).
        """
        hp = self._hparams
        with nn.parameter_scope('conv_norm_postnet'):
            out = inputs  # (B, T/r, n_mels * r)
            in_channels = [hp.postnet_embedding_dim] * \
                hp.postnet_n_convolutions + [hp.n_mels * hp.r]
            for i, channels in enumerate(in_channels):
                with nn.parameter_scope(f'filter_{i}'):
                    w_init_gain = 'affine' if i == len(
                        in_channels) - 1 else 'tanh'
                    out = conv_norm(
                        out, out_channels=channels, kernel_size=hp.postnet_kernel_size,
                        stride=1, padding=(hp.postnet_kernel_size - 1) // 2, bias=False,
                        dilation=1, w_init_gain=w_init_gain, scope='conv_norm', channel_last=True
                    )  # (B, T, channels)
                    out = PF.batch_normalization(
                        out, batch_stat=self.training, axes=[2])
                    if i < len(in_channels) - 1:
                        out = F.tanh(out)
                    if self.training:
                        out = F.dropout(out, 0.5)
        return out  # (B, T/r, n_mels * r)


class Tacotron2(Module):
    r"""An implementation of Tacotron 2.

    Args:
        hparams (HPrams): A container containing all hyper-parameters.

    References:
        [1] Shen et al., 2018. Natural tts synthesis by conditioning wavenet on mel spectrogram predictions.
        In ICASSP (pp. 4779-4783).
    """

    def __init__(self, hparams):
        self._hparams = hparams
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = PostNet(hparams)

    def call(self, char_inputs, mel_inputs=None):
        r"""Return mel-spectrograms, spectrograms, and attention weights.

        Args:
            char_inputs (nn.Variable): Inputs containing indices of characters.
                This has a shape of(B, Tx).

        Returns:
            nn.Variable: The synthetic mel-spectrograms of shape (B, T_y/r, n_mels*r).
            nn.Variable: The synthetic mel-spectrograms of shape (B, T_y/r, n_mels*r) after postnet.
            nn.Variable: The attention matrix of shape (B, Tx, Ty).
        """
        with nn.parameter_scope('encoder'):
            encoder_outputs = self.encoder(char_inputs)  # (Tx, B, C=512)

        with nn.parameter_scope('decoder'):
            encoder_outputs = F.transpose(encoder_outputs, (1, 0, 2))
            mel_outputs, gate_outputs, alignments = self.decoder(
                encoder_outputs, mel_inputs)

        with nn.parameter_scope('post_net'):
            mel_outputs_postnet = self.postnet(mel_outputs) + mel_outputs

        return mel_outputs, mel_outputs_postnet, gate_outputs, alignments
