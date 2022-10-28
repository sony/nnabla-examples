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

import math
from config import ModelConfig
import numpy as np
import click

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.initializer as I
from nnabla.parameter import get_parameter_or_create
from nnabla.logger import logger

from layers import *
from utils import Shape4D, context_scope, force_float


class ResidualBlock(object):
    def __init__(self,
                 out_channels=None,
                 scale_shift_norm=False,
                 dropout=0,
                 conv_shortcut=False,
                 channel_last=False,
                 rescale_skip=False):
        self.out_channels = out_channels
        self.scale_shift_norm = scale_shift_norm
        self.dropout = dropout
        self.conv_shortcut = conv_shortcut
        self.channel_last = channel_last
        self.rescale_skip = rescale_skip

    def in_layers(self, x):
        h = group_norm(x, name='norm_in',
                       channel_axis=3 if self.channel_last else 1)
        h = nonlinearity(h)
        h = conv(h, self.out_channels, name='conv_in',
                 channel_last=self.channel_last)

        return h

    def emb_layers(self, emb):
        out_channels = self.out_channels
        if self.scale_shift_norm:
            out_channels *= 2

        return nin(emb, out_channels, name="emb_proj", channel_last=self.channel_last)

    def out_layers(self, h, emb):
        if self.scale_shift_norm:
            scale, shift = chunk(
                emb, num_chunk=2, axis=3 if self.channel_last else 1)
            h = group_norm(
                h, name="norm_out", channel_axis=3 if self.channel_last else 1) * (scale + 1) + shift
        else:
            h += emb
            h = group_norm(h, name="norm_out",
                           channel_axis=3 if self.channel_last else 1)

        h = nonlinearity(h)

        if self.dropout > 0:
            h = F.dropout(h, p=self.dropout)

        h = conv(h, self.out_channels, name="conv_out",
                 zeroing_w=True, channel_last=self.channel_last)

        return h

    def shortcut(self, x):
        x_shape = Shape4D(x.shape, channel_last=self.channel_last)

        if self.out_channels == x_shape.c:
            return x
        elif self.conv_shortcut:
            return conv(x, self.out_channels, name="conv_shortcut", channel_last=self.channel_last)

        return nin(x, self.out_channels, name="conv_shortcut", channel_last=self.channel_last)


    def __call__(self, x, temb, name):
        x_shape = Shape4D(x.shape, channel_last=self.channel_last)
        if self.out_channels is None:
            self.out_channels = x_shape.c

        with nn.parameter_scope(name):
            # first block
            h = self.in_layers(x)

            # embed
            emb = self.emb_layers(temb)

            # second block
            h = self.out_layers(h, emb)

            # skip connection
            if self.rescale_skip:
                out = (h + self.shortcut(x)) * (2 ** -0.5)
            else:
                out = h + self.shortcut(x)

        return out


class ResidualBlockResampleBase(ResidualBlock):
    def resample(self, x, name):
        raise NotImplementedError("Resample method must be implemented.")

    def in_layers_with_resampling(self, x):
        h = group_norm(x, name='norm_in',
                       channel_axis=3 if self.channel_last else 1)
        h = nonlinearity(h)

        # apply downsampling to both x and h.
        x_res = self.resample(x, name="downsample_x")
        h = self.resample(h, name="downsample_h")

        h = conv(h, self.out_channels, name='conv_in',
                 channel_last=self.channel_last)

        return x_res, h

    def __call__(self, x, temb, name):
        x_shape = Shape4D(x.shape, channel_last=self.channel_last)
        if self.out_channels is None:
            self.out_channels = x_shape.c

        with nn.parameter_scope(name):
            # first block
            x_res, h = self.in_layers_with_resampling(x)

            # embed
            emb = self.emb_layers(temb)

            # second block
            h = self.out_layers(h, emb)

            # skip connection
            if self.rescale_skip:
                out = (h + self.shortcut(x_res)) / math.sqrt(2)
            else:
                out = h + self.shortcut(x_res)

        return out


class ResidualBlockDown(ResidualBlockResampleBase):
    def resample(self, x, name):
        return downsample(x, name, with_conv=False, channel_last=self.channel_last)


class ResidualBlockUp(ResidualBlockResampleBase):
    def resample(self, x, name):
        return upsample(x, name, with_conv=False, channel_last=self.channel_last)


# Attention
def self_attention(x,
                   name,
                   *,
                   cond=None,
                   num_heads=4,
                   num_head_channels=None,
                   channel_last=False):

    assert len(
        x.shape) == 4, "self_attention only supports 4D tensor for an input."
    B, C, H, W = Shape4D(
        x.shape, channel_last=channel_last).get_as_tuple("bchw")

    # apply normalization and projection for inputs
    with nn.parameter_scope(name):
        # Get query, key, value
        h = group_norm(x, name="norm", channel_axis=3 if channel_last else 1)
        qkv = nin(h, 3 * C, name="qkv", channel_last=channel_last)

        # 4D -> 3D
        if channel_last:
            # (B, H, W, 3C) -> (B, HW, 3C)
            qkv = F.reshape(qkv, (B, H * W, 3 * C))
        else:
            # (B, 3C, H, W) -> (B, 3C, HW)
            qkv = F.reshape(qkv, (B, 3 * C, H * W))

        q, k, v = chunk(qkv, num_chunk=3, axis=2 if channel_last else 1)

        # if cond is given, concat it to k and v along L axis.
        if cond is not None:
            # Assume text of shape (B, N, C), where B is batch, N is # tokens, and C is channel.
            assert len(
                cond.shape) == 3, "A condition for self_attention should be a 3D tensor."
            with nn.parameter_scope("condition"):
                # Imagen paper says layer_norm performs better for condition
                cond = layer_norm(cond, name="norm")
                cond = nin(cond, 2 * C, name="kv_cond",
                           channel_last=channel_last)

            kc, vc = chunk(cond, num_chunk=2, axis=2 if channel_last else 1)
            k = F.concatenate(k, kc, axis=1 if channel_last else 2)
            v = F.concatenate(v, vc, axis=1 if channel_last else 2)

        qkv_attention = QKVAttention(num_heads=num_heads,
                                     num_head_channels=num_head_channels,
                                     channel_last=channel_last)
        out = qkv_attention(q, k, v)

        # 3D -> 4D
        if channel_last:
            # (B, HW, C) -> (B, H, W, C)
            out = F.reshape(out, (B, H, W, C))
        else:
            # (B, C, HW) -> (B, C, H, W)
            out = F.reshape(out, (B, C, H, W))

        # output projection
        out = nin(out, C,
                  name='proj_out',
                  zeroing_w=True,
                  channel_last=channel_last)

    assert out.shape == x.shape
    return out + x


def cross_attention(x,
                    cond,
                    name,
                    *,
                    num_heads=4,
                    num_head_channels=None,
                    channel_last=False):

    assert len(
        x.shape) == 4, "corss_attention only supports 4D tensor for an input."
    B, C, H, W = Shape4D(x, channel_last=channel_last).get_as_tuple("bchw")

    with nn.parameter_scope(name):
        # Get query, key, value
        h = group_norm(x, name="norm", channel_axis=3 if channel_last else 1)
        q = nin(h, C, name="q", channel_last=channel_last)

        # Assume text of shape (B, N, C), where B is batch, N is # tokens, and C is channel.
        assert len(
            cond.shape) == 3, "A condition for cross_attention should be a 3D tensor."
        with nn.parameter_scope("condition"):
            # Imagen paper says layer_norm performs better for condition
            cond = layer_norm(cond, name="norm")
            cond = nin(cond, 2 * C, name="kv", channel_last=channel_last)

        k, v = chunk(cond, num_chunk=2, axis=2 if channel_last else 1)

        # 4D -> 3D
        if channel_last:
            # (B, H, W, 3C) -> (B, HW, 3C)
            q = F.reshape(q, (B, H * W, 3 * C))
        else:
            # (B, 3C, H, W) -> (B, 3C, HW)
            q = F.reshape(q, (B, 3 * C, H * W))

        qkv_attention = QKVAttention(num_heads=num_heads,
                                     num_head_channels=num_head_channels,
                                     channel_last=channel_last)
        out = qkv_attention(q, k, v)

        # 3D -> 4D
        if channel_last:
            # (B, HW, C) -> (B, H, W, C)
            out = F.reshape(out, (B, H, W, C))
        else:
            # (B, C, HW) -> (B, C, H, W)
            out = F.reshape(out, (B, C, H, W))

        # output projection
        out = nin(out, C,
                  name='proj_out',
                  zeroing_w=True,
                  channel_last=channel_last)

    assert out.shape == x.shape
    return out + x


def attention_pooling_1d(x,
                         name,
                         *,
                         num_heads=4,
                         num_head_channels=None,
                         channel_last=False,
                         keepdims=False):
    # adapted from https://github.com/openai/CLIP/blob/main/clip/model.py

    # get shape info
    assert len(x.shape) == 3, \
        "attention_pooling_1d only supports 3D tensor for an input."
    # (B, L, C) if channel last else (B, C, L)
    c_axis = 2 if channel_last else 1
    l_axis = 3 ^ c_axis # 3 XOR {1, 2} = {2, 1}
    C = x.shape[c_axis]
    L = x.shape[l_axis]

    with nn.parameter_scope(name):
        # compute mean. 
        # The mean must be concatenated before `x`` so as to support variable length.
        mean_x = F.mean(x, axis=l_axis, keepdims=True)
        h = F.concatenate(mean_x, x, axis=l_axis)

        # positional encoding
        # +1 in `L + 1` is for mean_emb
        pos_emb = get_parameter_or_create("pos_embed",
                                          shape=(L + 1, C) if channel_last else (C, L),
                                          initializer=I.NormalInitializer(C ** -0.5))
        h += F.reshape(pos_emb, (1, ) + pos_emb.shape)

        # apply layer norm before attention
        h = layer_norm(h, "ln_pre")

        # qkv projection
        # compute q from mean vector
        q = nin(h[:, :1], C, name="q", channel_last=channel_last)

        # comput kv from the original input
        kv = nin(h[:, 1:], 2 * C, name="kv", channel_last=channel_last)
        k, v = chunk(kv, num_chunk=2, axis=c_axis)

        qkv_attention = QKVAttention(num_heads=num_heads,
                                     num_head_channels=num_head_channels,
                                     channel_last=channel_last)
        out = qkv_attention(q, k, v)

        # output projection
        out = nin(out, C,
                  name='proj_out',
                  zeroing_w=True,
                  channel_last=channel_last)

        # check output shape
        output_shape = list(x.shape)
        output_shape[l_axis] = 1
        assert out.shape == tuple(output_shape)
    
        if not keepdims:
            # squeze L axis 
            out = out[:, 0, :]
        
    return out



class QKVAttention(object):
    def __init__(self, num_heads=4, num_head_channels=None, channel_last=False):
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.channel_last = channel_last

    def _validate_input_shape(self, q, k, v):
        # check all inputs are 3D tensor.
        assert len(q.shape) == len(k.shape) == len(v.shape) == 3, \
            "inputs for QKVAttention must be a 3D tensor."

        # define axes
        b_axis = 0
        c_axis = 2 if self.channel_last else 1
        l_axis = 3 ^ c_axis # 3 XOR {1, 2} = {2, 1}

        # check batch size
        assert q.shape[b_axis] == k.shape[b_axis] == v.shape[b_axis], \
            "All inputs must have the same batch size."
        
        # check spacial size
        assert k.shape[l_axis] == v.shape[l_axis], \
            "k and v must have the same length."

        # check channel size
        # strong assumuption that q, k, and v have the same channel dims.
        assert q.shape[c_axis] == k.shape[c_axis] == v.shape[c_axis], \
            "All inputs must have the same channel dims."

        C = q.shape[c_axis]

        # check attention configuration
        if self.num_head_channels is not None:
            assert isinstance(self.num_head_channels, int), \
                f"num_head_channels must be an interger but {type(self.num_head_channels)}"
            assert C % self.num_head_channels == 0, \
                f"input channels (= {C}) is not divisible by num_head_channels (= {self.num_head_channels})"

            self.num_heads = C // self.num_head_channels

        assert self.num_heads is not None and C % self.num_heads == 0, \
            f"input channels (= {C}) is not divisible by num_heads (= {self.num_heads})"

        self.num_head_channels = C // self.num_heads

    def __call__(self, q, k, v):
        """
        Compute attention based on given q, k, and v.
        """
        # check input shape
        self._validate_input_shape(q, k, v)

        if not self.channel_last:
            # make qkv channel-last tensor once so as to make the following code simple.
            # (B, C, L) -> (B, L, C)
            q = F.transpose(q, (0, 2, 1))
            k = F.transpose(k, (0, 2, 1))
            v = F.transpose(v, (0, 2, 1))

        # L may vary for q and (k, v).
        B, _, C = q.shape

        # scale is applied both q and k to avoid large value in dot op.
        scale = 1 / math.sqrt(math.sqrt(self.num_head_channels))

        q = F.reshape(q * scale, (B * self.num_heads, q.shape[1], -1))
        k = F.reshape(k * scale, (B * self.num_heads, k.shape[1], -1))
        v = F.reshape(v, (B * self.num_heads, v.shape[1], -1))

        # create attention weight
        # (B * num_heads, L (for q), L (for k))
        w = F.batch_matmul(q, k, transpose_b=True)
        with context_scope("float"):
            w = F.softmax(w, axis=-1)  # take softmax for each query

        # attention
        a = F.reshape(F.batch_matmul(w, v), (B, -1, C))

        if not self.channel_last:
            # (B, L, C) -> (B, C, L)
            a = F.transpose(a, (0, 2, 1))

        return a


class UNet(object):
    def __init__(self, conf: ModelConfig):
        # check args
        assert hasattr(conf.channel_mult, "__iter__"), \
            f"channel_mult must be an iterable object, but '{type(conf.channel_mult)}' is given."

        # (2022/07/14) omegaconf doesn't support Union for typing. Thus, cannot overwrite conf.num_res_blocks.
        if isinstance(conf.num_res_blocks, int):
            self.num_res_blocks = [
                conf.num_res_blocks for _ in range(len(conf.channel_mult))]
        else:
            assert hasattr(conf.num_res_blocks, "__iter__") \
                   and len(conf.num_res_blocks) == len(conf.channel_mult), \
                    f"length of num_res_blocks and channel_mult must be the same (num_res_block: {conf.num_res_blocks}, channel_mult: {conf.channel_mult})"
            self.num_res_blocks = conf.num_res_blocks

        self.conf: ModelConfig = conf
        self.emb_dims = 4 * self.conf.base_channels
        self.use_mixed_precision = conf.use_mixed_precision

    # condition branch
    def concat_input_cond(self, x, input_cond):
        assert isinstance(input_cond, (nn.Variable, nn.NdArray)
                          ), "input_cond must be nn.Variable or nn,NdArray."
        assert x.shape[0] == input_cond.shape[0], "batch size must be the same between x and input_cond"

        return F.concatenate(x,
                             interp_like(input_cond, x,
                                         channel_last=self.conf.channel_last),
                             axis=3 if self.conf.channel_last else 1)

    @staticmethod
    def masking(x, mask_prob, axis):
        if mask_prob == 0:
            return x

        mask_shape = [1 for _ in range(len(x.shape))]
        mask_shape[axis] = x.shape[axis]

        if mask_prob == 1:
            mask = F.constant(shape=mask_shape)
        else:
            mask = F.rand_binomial(p=1-mask_prob,
                                   shape=mask_shape)
        return x * mask
    
    def global_embedding_to_4d(self, emb):
        if self.conf.channel_last:
            return F.reshape(emb, (emb.shape[0], 1, 1, emb.shape[1]))
        
        return F.reshape(emb, emb.shape + (1, 1))

    def embedding_projection(self, emb, mode: str, to_4d: bool = True):
        # reshape to use conv rather than affine
        if to_4d:
            emb = self.global_embedding_to_4d(emb)

        # post process
        mode = mode.lower()
        if mode == "simple":
            # pass emb as is
            return emb
        elif mode == "mlp":
            # linear transforms
            emb = nin(emb, self.emb_dims, name='dense0',
                      channel_last=self.conf.channel_last)
            emb = nonlinearity(emb)

            emb = nin(emb, self.emb_dims, name='dense1',
                      channel_last=self.conf.channel_last)
            return emb
        elif mode == "ln_mlp":
            # apply layer norm first.
            emb = layer_norm(emb, name="ln_0")

            # fall back to mlp
            return self.embedding_projection(emb, mode="mlp", to_4d=False)

        raise NotImplementedError(
            f"embedding projection mode `{mode}` is not supported.")

    def timestep_embedding(self, t, name=None):
        if name is None:
            name = 'timestep_embedding'

        with nn.parameter_scope(name):
            # sinusoidal embedding
            emb = sinusoidal_embedding(t, self.conf.base_channels)

            emb = self.embedding_projection(emb, mode="mlp")

        return emb

    def class_embedding(self, class_label, drop_rate):
        assert len(class_label.shape) == 1, \
             f"Invalid shape for class_label: {class_label.shape}."
        assert 0 <= drop_rate <= 1, "drop_rate must be in the range of [0, 1]."

        with nn.parameter_scope("class_embedding"):
            emb = PF.embed(inp=class_label,
                           n_inputs=self.conf.num_classes,
                           n_features=self.emb_dims,
                           initializer=I.NormalInitializer())  # align init with pytorch

            # dropout for unconditional generation
            emb = self.masking(emb,
                               mask_prob=drop_rate,
                               axis=0)

            emb = self.embedding_projection(
                emb, mode=self.conf.class_cond_emb_type)

        return emb

    def text_embedding(self, text_emb, drop_rate):
        # assume text_emb is already embeded by a pretrained model like t5.
        # A shape should be (B, L, C), where B is batch, L is length, and C is emb dim.
        assert len(text_emb.shape) == 3, \
             f"Invalid shape for text_emb: {text_emb.shape}."
        assert 0 <= drop_rate <= 1, "drop_rate must be in the range of [0, 1]."

        with nn.parameter_scope("text_embedding"):
            # dropout for unconditional generation
            emb = self.masking(text_emb,
                               mask_prob=drop_rate,
                               axis=0)

            # apply mlp to text embedding
            assert self.conf.channel_last, \
                "text_embedding assumes channel_last layout."
            emb = self.embedding_projection(emb, mode="ln_mlp", to_4d=False)

            # apply pooling along sequence to get global condition
            emb_pooled = attention_pooling_1d(emb, 
                                              name="attention_pooling", 
                                              num_heads=self.conf.num_attention_heads,
                                              num_head_channels=self.conf.num_attention_head_channels,
                                              channel_last=self.conf.channel_last,
                                              keepdims=False)

            # to 4d
            emb_pooled = self.global_embedding_to_4d(emb_pooled)

        return emb, emb_pooled

    # according to "guided diffusion", all computation for condition vectors seems to be done with float32.
    @force_float
    def condition_processing(self,
                             x,
                             t,
                             *,
                             input_cond=None,
                             input_cond_aug_timestep=None,
                             class_label=None,
                             text_emb=None,
                             cond_drop_rate=0):
        # timestep t
        emb = self.timestep_embedding(t)

        # input condition. Typically, low resolution image for upsampler.
        if input_cond is not None:
            x = self.concat_input_cond(x, input_cond)

            # gaussian conditioning timestep if applied.
            if self.conf.noisy_low_res:
                assert isinstance(input_cond_aug_timestep, nn.Variable), \
                    "input_cond_aug_timestep must be an instance of nn.Variable"

                emb += self.timestep_embedding(input_cond_aug_timestep,
                                                "gaussian_conditioning_timestep_embedding")

        # class condition
        if self.conf.class_cond:
            assert class_label is not None, "class_label must be nn.Variable or nn.NdArray"
            emb += self.class_embedding(class_label, cond_drop_rate)

        # text condition
        emb_seq = None
        if self.conf.text_cond:
            assert text_emb is not None, "text_emb must be passed"
            emb_seq, text_emb_pooled = self.text_embedding(
                text_emb, cond_drop_rate)
            emb += text_emb_pooled

        emb = nonlinearity(emb)

        return emb, emb_seq

    # main branch
    def resblock_with_attention(self, h, emb, emb_seq, out_channels, name):
        with nn.parameter_scope(name):
            block = ResidualBlock(out_channels,
                                  scale_shift_norm=self.conf.scale_shift_norm,
                                  dropout=self.conf.dropout,
                                  channel_last=self.conf.channel_last,
                                  rescale_skip=self.conf.resblock_rescale_skip)

            h = block(h, emb, "res_block")

            res = Shape4D(
                h.shape, channel_last=self.conf.channel_last).get_as_tuple("h")
            if self.conf.attention_resolutions is not None \
                    and res in self.conf.attention_resolutions:
                if self.conf.attention_type == "self_attention":
                    h = self_attention(h,
                                       name="attention",
                                       cond=emb_seq,
                                       num_heads=self.conf.num_attention_heads,
                                       num_head_channels=self.conf.num_attention_head_channels,
                                       channel_last=self.conf.channel_last)
                elif self.conf.attention_type == "cross_attention":
                    h = cross_attention(h, emb_seq,
                                        name="cross_attention",
                                        num_heads=self.conf.num_attention_heads,
                                        num_head_channels=self.conf.num_attention_head_channels,
                                        channel_last=self.conf.channel_last)
                else:
                    raise ValueError(
                        f"'{self.conf.attention_type}' is not supported for attention type.")

        return h

    def downsample_blocks(self, h, emb, emb_seq, out_channels, level, down, num_res_block):
        hs = []
        with nn.parameter_scope(f"block_{level}"):
            for i in range(num_res_block):
                h = self.resblock_with_attention(
                    h, emb, emb_seq, out_channels, name=f"resblock_{i}")
                hs.append(h)

        if down:
            if self.conf.resblock_resample:
                resblock_down = ResidualBlockDown(out_channels=out_channels,
                                                  scale_shift_norm=self.conf.scale_shift_norm,
                                                  dropout=self.conf.dropout,
                                                  channel_last=self.conf.channel_last,
                                                  rescale_skip=self.conf.resblock_rescale_skip)
                h = resblock_down(h, emb, name=f"downsample_{level}")
            else:
                h = downsample(h,
                               name=f"downsample_{level}",
                               with_conv=self.conf.conv_resample,
                               channel_last=self.conf.channel_last)

            hs.append(h)

        return hs

    def upsample_blocks(self, h, emb, emb_seq, hs_down, out_channels, level, up, num_res_block):
        with nn.parameter_scope(f"output_{level}"):
            for i in range(num_res_block + 1):
                # concat skip
                skip = hs_down.pop()
                h = F.concatenate(h, skip,
                                  axis=3 if self.conf.channel_last else 1)
                h = self.resblock_with_attention(
                    h, emb, emb_seq, out_channels, name=f"resblock_{i}")

        if up:
            if self.conf.resblock_resample:
                resblock_up = ResidualBlockUp(out_channels=out_channels,
                                              scale_shift_norm=self.conf.scale_shift_norm,
                                              dropout=self.conf.dropout,
                                              channel_last=self.conf.channel_last,
                                              rescale_skip=self.conf.resblock_rescale_skip)
                h = resblock_up(h, emb, name=f"upsample_{level}")
            else:
                h = upsample(h,
                             name=f"upsample_{level}",
                             with_conv=self.conf.conv_resample,
                             channel_last=self.conf.channel_last)

        return h

    def middle_block(self, h, emb, emb_seq):
        ch = h.shape[-1 if self.conf.channel_last else 1]
        block = ResidualBlock(ch,
                              scale_shift_norm=self.conf.scale_shift_norm,
                              dropout=self.conf.dropout,
                              channel_last=self.conf.channel_last,
                              rescale_skip=self.conf.resblock_rescale_skip)

        h = block(h, emb, name="resblock_0")

        res = Shape4D(h.shape, self.conf.channel_last).get_as_tuple("h")
        if self.conf.attention_resolutions is not None \
                and res in self.conf.attention_resolutions:
            if self.conf.attention_type == "self_attention":
                h = self_attention(h,
                                   name="attention",
                                   cond=emb_seq,
                                   num_heads=self.conf.num_attention_heads,
                                   num_head_channels=self.conf.num_attention_head_channels,
                                   channel_last=self.conf.channel_last)
            elif self.conf.attention_type == "cross_attention":
                h = cross_attention(h, emb_seq,
                                    name="cross_attention",
                                    num_heads=self.conf.num_attention_heads,
                                    num_head_channels=self.conf.num_attention_head_channels,
                                    channel_last=self.conf.channel_last)
            else:
                raise ValueError(
                    f"'{self.conf.attention_type}' is not supported for attention type.")

        h = block(h, emb, name="resblock_1")

        return h

    def output_block(self, h):
        h = group_norm(h,
                       name="last_norm",
                       channel_axis=3 if self.conf.channel_last else 1)
        h = nonlinearity(h)
        h = conv(h,
                 self.conf.output_channels,
                 name="last_conv",
                 zeroing_w=True,
                 channel_last=self.conf.channel_last)

        return h

    # forward definition
    def __call__(self,
                 x,
                 t,
                 *,
                 name=None,
                 **model_kwargs):
        ch = self.conf.base_channels
        with context_scope("half" if self.use_mixed_precision else "float"), nn.parameter_scope('UNet' if name is None else name):
            # condition
            emb, emb_seq = self.condition_processing(x, t, **model_kwargs)

            # first convolution
            if self.conf.channel_last:
                # todo: If we allow tf32, might be better to apply pad even if chennel_first.
                # But, to do that, we have to care obsolete parameters.
                x = pad_for_faster_conv(x, channel_last=self.conf.channel_last)
            h = conv(x, ch, name="first_conv",
                     channel_last=self.conf.channel_last)

            hs = [h]
            # downsample block
            with nn.parameter_scope("downsample_block"):
                for level, (mult, num_res_block) in enumerate(zip(self.conf.channel_mult,
                                                                  self.num_res_blocks)):
                    # downsample to lower resolution except last
                    is_last_block = level == len(self.conf.channel_mult) - 1

                    # apply resblock and attention for this resolution
                    outs = self.downsample_blocks(h,
                                                  emb,
                                                  emb_seq,
                                                  ch * mult,
                                                  level=level,
                                                  down=not is_last_block,
                                                  num_res_block=num_res_block)
                    hs += outs
                    h = outs[-1]

            # middle block
            with nn.parameter_scope("middle_block"):
                h = self.middle_block(h, emb, emb_seq)

            # upsample block
            with nn.parameter_scope("upsample_block"):
                for level, (mult, num_res_block) in enumerate(zip(reversed(self.conf.channel_mult),
                                                                  reversed(self.num_res_blocks))):
                    # upsample to larger resolution except last
                    is_last_block = level == len(self.conf.channel_mult) - 1

                    # apply resblock and attention for this resolution
                    h = self.upsample_blocks(h,
                                             emb,
                                             emb_seq,
                                             hs,
                                             ch * mult,
                                             level=level,
                                             up=not is_last_block,
                                             num_res_block=num_res_block)

            assert len(hs) == 0

            # output block
            with nn.parameter_scope("output_block"):
                out = self.output_block(h)

            out_shape = Shape4D(x.shape, channel_last=self.conf.channel_last)
            out_shape.c = self.conf.output_channels
            assert Shape4D(out.shape,
                           channel_last=self.conf.channel_last) == out_shape

        return out


class EfficientUNet(UNet):
    def __init__(self, *args, **kwargs):
        super(EfficientUNet, self).__init__(*args, **kwargs)

    def downsample_blocks(self, h, emb, emb_seq, out_channels, level, num_res_block):
        # 1. downsample (strided conv) -> 2. resblock x n -> 3. attention
        # for skip connection, use only the last output (not intermediate layers in each resolusion block)
        hs = []

        # 1. downsample
        if self.conf.resblock_resample:
            logger.warning(
                "Downsample by residual block. This is *not* a default setting for Efficient-UNet.")
            h = ResidualBlockDown(out_channels=out_channels,
                                  scale_shift_norm=self.conf.scale_shift_norm,
                                  dropout=self.conf.dropout,
                                  channel_last=self.conf.channel_last,
                                  rescale_skip=self.conf.resblock_rescale_skip)(h, emb, name=f"downsample_{level}")
        else:
            h = downsample(h,
                           name=f"downsample_{level}",
                           with_conv=self.conf.conv_resample,
                           channel_last=self.conf.channel_last)

        with nn.parameter_scope(f"block_{level}"):
            # 2. resblock x n
            block = ResidualBlock(out_channels,
                                  scale_shift_norm=self.conf.scale_shift_norm,
                                  dropout=self.conf.dropout,
                                  channel_last=self.conf.channel_last,
                                  rescale_skip=self.conf.resblock_rescale_skip)
            for i in range(num_res_block):
                h = block(h, emb, f"resblock_{i}")
                hs.append(h)

            # 3. attention
            res = Shape4D(
                h.shape, channel_last=self.conf.channel_last).get_as_tuple("h")
            if self.conf.attention_resolutions is not None \
                    and res in self.conf.attention_resolutions:

                if self.conf.attention_type == "self_attention":
                    h = self_attention(h,
                                       name="attention",
                                       cond=emb_seq,
                                       num_heads=self.conf.num_attention_heads,
                                       num_head_channels=self.conf.num_attention_head_channels,
                                       channel_last=self.conf.channel_last)
                elif self.conf.attention_type == "cross_attention":
                    h = cross_attention(h, emb_seq,
                                        name="cross_attention",
                                        num_heads=self.conf.num_attention_heads,
                                        num_head_channels=self.conf.num_attention_head_channels,
                                        channel_last=self.conf.channel_last)
                else:
                    raise ValueError(
                        f"'{self.conf.attention_type}' is not supported for attention type.")

                hs.append(h)

        return hs

    def upsample_blocks(self, h, emb, emb_seq, hs_down, out_channels, level, num_res_block):
        # 1. skip connection -> 2. resblock x n -> 3. attention -> 4. upsample

        # 1. skip connection
        # (2022/10/20) follow guided-diffusion's skip connection
        def skip_connection(u, v, rescale_skip=False):
            return F.concatenate(u, 
                                 v * (2 ** -0.5) if rescale_skip else v,
                                 axis=3 if self.conf.channel_last else 1)

        with nn.parameter_scope(f"output_{level}"):
            # 2. resblock x n
            block = ResidualBlock(out_channels,
                                  scale_shift_norm=self.conf.scale_shift_norm,
                                  dropout=self.conf.dropout,
                                  channel_last=self.conf.channel_last,
                                  rescale_skip=self.conf.resblock_rescale_skip)
            for i in range(num_res_block):
                h = skip_connection(h, hs_down.pop())
                h = block(h, emb, f"resblock_{i}")

            # 3. attention
            res = Shape4D(
                h.shape, channel_last=self.conf.channel_last).get_as_tuple("h")
            if self.conf.attention_resolutions is not None and res in self.conf.attention_resolutions:
                h = skip_connection(h, hs_down.pop())

                if self.conf.attention_type == "self_attention":
                    h = self_attention(h,
                                       name="attention",
                                       cond=emb_seq,
                                       num_heads=self.conf.num_attention_heads,
                                       num_head_channels=self.conf.num_attention_head_channels,
                                       channel_last=self.conf.channel_last)
                elif self.conf.attention_type == "cross_attention":
                    h = cross_attention(h, emb_seq,
                                        name="cross_attention",
                                        num_heads=self.conf.num_attention_heads,
                                        num_head_channels=self.conf.num_attention_head_channels,
                                        channel_last=self.conf.channel_last)
                else:
                    raise ValueError(
                        f"'{self.conf.attention_type}' is not supported for attention type.")
        # 4. upsample
        if self.conf.resblock_resample:
            logger.warning(
                "Upsample by residual block. This is *not* a default setting for Efficient-UNet.")
            h = ResidualBlockUp(out_channels=out_channels,
                                scale_shift_norm=self.conf.scale_shift_norm,
                                dropout=self.conf.dropout,
                                channel_last=self.conf.channel_last,
                                rescale_skip=self.conf.resblock_rescale_skip)(h, emb, name=f"upsample_{level}")
        else:
            h = upsample(h,
                         name=f"upsample_{level}",
                         with_conv=self.conf.conv_resample,
                         channel_last=self.conf.channel_last)

        return h

    def output_block(self, h):
        return nin(h,
                   self.conf.output_channels,
                   name="last_conv",
                   zeroing_w=True, # needed?
                   channel_last=self.conf.channel_last)

    def __call__(self,
                 x,
                 t,
                 *,
                 name=None,
                 **model_kwargs):
        ch = self.conf.base_channels
        with context_scope("half" if self.use_mixed_precision else "float"), nn.parameter_scope('E-UNet' if name is None else name):
            # condition
            emb, emb_seq = self.condition_processing(x, t, **model_kwargs)

            if self.conf.channel_last:
                # todo: If we allow tf32, might be better to apply pad even if chennel_first.
                # But, to do that, we have to care obsolete parameters.
                x = pad_for_faster_conv(x, channel_last=self.conf.channel_last)

            h = conv(x, ch, name="first_conv",
                     channel_last=self.conf.channel_last)

            hs = []
            # downsample block
            with nn.parameter_scope("downsample_block"):
                for level, (mult, num_res_block) in enumerate(zip(self.conf.channel_mult, self.num_res_blocks)):
                    # apply resblock and attention for this resolution
                    outs = self.downsample_blocks(h,
                                                  emb,
                                                  emb_seq,
                                                  ch * mult,
                                                  level=level,
                                                  num_res_block=num_res_block)
                    hs += outs
                    h = outs[-1]

            # middle block
            with nn.parameter_scope("middle_block"):
                h = self.middle_block(h, emb, emb_seq)

            # upsample block
            with nn.parameter_scope("upsample_block"):
                for level, (mult, num_res_block) in enumerate(zip(reversed(self.conf.channel_mult), reversed(self.num_res_blocks))):
                    # apply resblock and attention for this resolution
                    h = self.upsample_blocks(h,
                                             emb,
                                             emb_seq,
                                             hs,
                                             ch * mult,
                                             level=level,
                                             num_res_block=num_res_block)

            assert len(hs) == 0

            # output block
            with nn.parameter_scope("output_block"):
                out = self.output_block(h)

            out_shape = Shape4D(x.shape, channel_last=self.conf.channel_last)
            out_shape.c = self.conf.output_channels
            assert Shape4D(
                out.shape, channel_last=self.conf.channel_last) == out_shape

        return out

# Functions below are for dubugging UNet class.


def test_simple_loop():
    nn.clear_parameters()

    x = nn.Variable.from_numpy_array(np.random.randn(10, 3, 128, 128))
    t = nn.Variable.from_numpy_array(np.random.randint(0, 100, (10, )))

    conf = ModelConfig(num_classes=1,
                       base_channels=128,
                       output_channels=3,
                       num_res_blocks=2,
                       attention_resolutions=(16, 8),
                       num_attention_heads=4,
                       channel_mult=(1, 1, 2, 2, 4, 4))

    unet = UNet(conf)
    y = unet(x, t)

    loss = F.mean(F.squared_error(y, x))

    import nnabla.solvers as S
    solver = S.Sgd()
    solver.set_parameters(nn.get_parameters())

    from tqdm import trange
    tr = trange(100)
    for i in tr:
        loss.forward(clear_no_need_grad=True)
        solver.zero_grad()
        loss.backward(clear_buffer=True)
        solver.update()

        tr.set_description(f"diff: {loss.d.copy():.5f}")


def dump(var):
    if isinstance(var, list):
        for x in var:
            dump(x)

        return

    arr = var.d
    mean = arr.mean()
    std = arr.std()
    abs_sum = np.abs(arr).sum()
    print("mean: {:-6.5g} std: {:-6.5g} abs_sum: {:-6.5g} shape: {}".format(mean,
          std, abs_sum, arr.shape))


@click.command()
@click.option("--loop/--no-loop", default=False)
@click.option("--intermediate/--no-intermediate", default=False)
@click.option("--config", default=None)
@click.option("--h5", default=None)
def test(loop, intermediate, config, h5):
    # This function is for a unit test of UNet.
    from nnabla.ext_utils import get_extension_context
    ctx = get_extension_context("cudnn")
    nn.set_default_context(ctx)

    from nnabla.logger import logger

    if loop:
        logger.info("Test Unet by simple training loop.")
        test_simple_loop()

    if intermediate:
        logger.info("Test intermediate values of Unet.")

        import os

        # check and load config
        conf = None
        if config is not None:
            assert os.path.exists(
                config), f"config file `{config}` is not found."
            logger.info(f"... with config={config} as followd:")

            from neu.yaml_wrapper import read_yaml
            conf = read_yaml(config)
            conf.dump()

        if h5 is not None:
            assert os.path.exists(h5), f"config file `{h5}` is not found."
            logger.info(f"... with h5={h5}")


if __name__ == "__main__":
    test()
