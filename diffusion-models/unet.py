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

import numpy as np
import click

import nnabla as nn
import nnabla.functions as F

from layers import *


class ResidualBlock(object):
    def __init__(self,
                 out_channels=None,
                 scale_shift_norm=False,
                 dropout=0,
                 conv_shortcut=False):
        self.out_channels = out_channels
        self.scale_shift_norm = scale_shift_norm
        self.dropout = dropout
        self.conv_shortcut = conv_shortcut

    def in_layers(self, x):
        h = normalize(x, name='norm_in')
        h = nonlinearity(h)
        h = conv(h, self.out_channels, name='conv_in')

        return h

    def emb_layers(self, emb):
        out_channels = self.out_channels
        if self.scale_shift_norm:
            out_channels *= 2

        return nin(emb, out_channels, name="emb_proj")

    def out_layers(self, h, emb):
        if self.scale_shift_norm:
            scale, shift = chunk(emb, num_chunk=2, axis=1)
            h = normalize(h, name="norm_out") * (scale + 1) + shift
        else:
            h += emb
            h = normalize(h, name="norm_out")

        h = nonlinearity(h)

        if self.dropout > 0:
            h = F.dropout(h, p=self.dropout)

        h = conv(h, self.out_channels, name="conv_out", zeroing_w=True)

        return h

    def shortcut(self, x):
        if self.out_channels == x.shape[1]:
            return x
        elif self.conv_shortcut:
            return conv(x, self.out_channels, name="conv_shortcut")
        else:
            return nin(x, self.out_channels, name="conv_shortcut")

    def __call__(self, x, temb, name):
        C = x.shape[1]
        if self.out_channels is None:
            self.out_channels = C

        with nn.parameter_scope(name):
            # first block
            h = self.in_layers(x)

            # embed
            emb = self.emb_layers(temb)

            # second block
            h = self.out_layers(h, emb)

            # add residual
            out = F.add2(h, self.shortcut(x), inplace=True)

        return out


def attn_block(x, name, num_heads=4, fix_parameters=False):
    """Multihead attention block"""
    B, C, H, W = x.shape

    with nn.parameter_scope(name):
        # Get query, key, value
        h = normalize(x, name="norm")
        # nin(3 * C) -> split is faster?
        q = nin(h, C, name="q")
        k = nin(h, C, name="k")
        v = nin(h, C, name="v")

        # Attention
        w = F.batch_matmul(F.reshape(q, (B * num_heads, -1, H * W)),
                           F.reshape(k, (B * num_heads, -1, H * W)), transpose_a=True)
        w = F.mul_scalar(w, int(C) ** (-0.5), inplace=True)

        assert w.shape == (B * num_heads, H * W, H * W)
        w = F.softmax(w, axis=-1)

        h = F.reshape(v, (B * num_heads, -1, H * W))
        h = F.batch_matmul(h, w)
        h = F.reshape(h, (B, C, H, W))

        # output projection
        h = nin(h, C, name='proj_out', zeroing_w=True)

    assert h.shape == x.shape
    return F.add2(h, x, inplace=True)


class UNet(object):
    def __init__(self,
                 num_classes,
                 model_channels,
                 output_channels,
                 num_res_blocks,
                 attention_resolutions,
                 attention_num_heads,
                 channel_mult=(1, 2, 4, 8),
                 dropout=0.,
                 scale_shift_norm=False,
                 conv_resample=True):
        self.num_classes = num_classes
        self.model_channels = model_channels
        self.output_channels = output_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.attention_num_heads = attention_num_heads
        self.channel_mult = channel_mult
        self.dropout = dropout
        self.scale_shift_norm = scale_shift_norm
        self.conv_resample = conv_resample

    def timestep_embedding(self, t):
        with nn.parameter_scope('timestep_embedding'):
            # sinusoidal embedding
            emb = sinusoidal_embedding(t, self.model_channels)

            # reshape to use conv rather than affine
            emb = F.reshape(emb, emb.shape + (1, 1))

            # linear transforms
            emb = nin(emb, self.model_channels * 4, name='dense0')
            emb = nonlinearity(emb)

            emb = nin(emb, self.model_channels * 4, name='dense1')

        return emb

    def resblock_with_attention(self, h, emb, out_channels, name):
        with nn.parameter_scope(name):
            block = ResidualBlock(out_channels,
                                  scale_shift_norm=self.scale_shift_norm,
                                  dropout=self.dropout)

            h = block(h, emb, f"res_block")

            res = h.shape[-1]
            if self.attention_resolutions is not None and res in self.attention_resolutions:
                h = attn_block(h, f"attention",
                               num_heads=self.attention_num_heads)

        return h

    def downsample_blocks(self, h, emb, out_channels, name):
        hs = []
        with nn.parameter_scope(name):
            for i in range(self.num_res_blocks):
                h = self.resblock_with_attention(
                    h, emb, out_channels, name=f"resblock_{i}")
                hs.append(h)

        return hs

    def upsample_blocks(self, h, emb, hs_down, out_channels, name):
        hs_up = []
        with nn.parameter_scope(name):
            for i in range(self.num_res_blocks + 1):
                # concat skip
                h = F.concatenate(h, hs_down.pop(), axis=1)
                h = self.resblock_with_attention(
                    h, emb, out_channels, name=f"resblock_{i}")
                hs_up.append(h)

        return hs_up

    def middle_block(self, h, emb):
        ch = h.shape[1]
        block = ResidualBlock(
            ch, scale_shift_norm=self.scale_shift_norm, dropout=self.dropout)

        h = block(h, emb, name="resblock_0")

        res = h.shape[-1]
        if self.attention_resolutions is not None and res in self.attention_resolutions:
            h = attn_block(h, f"attention", num_heads=self.attention_num_heads)

        h = block(h, emb, name="resblock_1")

        return h

    def output_block(self, h):
        h = normalize(h, "last_norm")
        h = nonlinearity(h)
        h = conv(h, self.output_channels, name="last_conv", zeroing_w=True)

        return h

    def get_intermediates(self, x, t, name=None):
        ret = dict()

        with nn.auto_forward(True):
            ch = self.model_channels
            with nn.parameter_scope('UNet' if name is None else name):
                h = conv(x, ch, name="first_conv")

                emb = self.timestep_embedding(t)
                ret["emb"] = emb
                emb = nonlinearity(emb)

                hs = [h]
                # downsample block
                with nn.parameter_scope("downsample_block"):
                    for level, mult in enumerate(self.channel_mult):
                        # apply resblock and attention for this resolution
                        outs = self.downsample_blocks(h, emb, ch * mult,
                                                      name=f"block_{level}")
                        hs += outs
                        h = outs[-1]

                        # downsample to lower resolution except last
                        if level < len(self.channel_mult) - 1:
                            h = downsample(h, name=f"downsample_{level}",
                                           with_conv=self.conv_resample)
                            hs.append(h)

                ret["down"] = hs.copy()

                # middle block
                with nn.parameter_scope("middle_block"):
                    h = self.middle_block(h, emb)

                ret["middle"] = h

                # upsample block
                hs_up = []
                with nn.parameter_scope("upsample_block"):
                    for level, mult in enumerate(reversed(self.channel_mult)):
                        # apply resblock and attention for this resolution
                        outs = self.upsample_blocks(h, emb, hs, ch * mult,
                                                    name=f"output_{level}")
                        h = outs[-1]

                        # downsample to lower resolution except last
                        if level < len(self.channel_mult) - 1:
                            h = upsample(h, name=f"upsample_{level}",
                                         with_conv=self.conv_resample)
                            outs.pop()
                            outs.append(h)

                        hs_up += outs
                assert len(hs) == 0

                ret["up"] = hs_up.copy()

                # output block
                with nn.parameter_scope("output_block"):
                    out = self.output_block(h)

                ret["out"] = out

                assert out.shape == x.shape[:1] + \
                    (self.output_channels, ) + x.shape[2:]

                return ret

    def __call__(self, x, t, name=None):
        ch = self.model_channels
        with nn.parameter_scope('UNet' if name is None else name):
            h = conv(x, ch, name="first_conv")
            emb = self.timestep_embedding(t)
            emb = nonlinearity(emb)

            hs = [h]
            # downsample block
            with nn.parameter_scope("downsample_block"):
                for level, mult in enumerate(self.channel_mult):
                    # apply resblock and attention for this resolution
                    outs = self.downsample_blocks(h, emb, ch * mult,
                                                  name=f"block_{level}")
                    hs += outs
                    h = outs[-1]

                    # downsample to lower resolution except last
                    if level < len(self.channel_mult) - 1:
                        h = downsample(h, name=f"downsample_{level}",
                                       with_conv=self.conv_resample)
                        hs.append(h)

            # middle block
            with nn.parameter_scope("middle_block"):
                h = self.middle_block(h, emb)

            # upsample block
            with nn.parameter_scope("upsample_block"):
                for level, mult in enumerate(reversed(self.channel_mult)):
                    # apply resblock and attention for this resolution
                    outs = self.upsample_blocks(h, emb, hs, ch * mult,
                                                name=f"output_{level}")
                    h = outs[-1]

                    # downsample to lower resolution except last
                    if level < len(self.channel_mult) - 1:
                        h = upsample(h, name=f"upsample_{level}",
                                     with_conv=self.conv_resample)

            assert len(hs) == 0

            # output block
            with nn.parameter_scope("output_block"):
                out = self.output_block(h)

            assert out.shape == x.shape[:1] + \
                (self.output_channels, ) + x.shape[2:]

            return out

# Functions below are for dubugging UNet class.


def test_simple_loop():
    nn.clear_parameters()

    x = nn.Variable.from_numpy_array(np.random.randn(10, 3, 128, 128))
    t = nn.Variable.from_numpy_array(np.random.randint(0, 100, (10, )))

    unet = UNet(num_classes=1, model_channels=128, output_channels=3,
                num_res_blocks=2,
                attention_resolutions=(16, 8),
                attention_num_heads=4,
                channel_mult=(1, 1, 2, 2, 4, 4))
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


def test_intermediate():
    import os

    os.environ["NNABLA_CUDNN_DETERMINISTIC"] = '1'

    nn.clear_parameters()

    x = nn.Variable.from_numpy_array(np.full((1, 3, 256, 256), 0.1))
    t = nn.Variable.from_numpy_array([803])

    unet = UNet(num_classes=1, model_channels=128, output_channels=3,
                num_res_blocks=3,
                attention_resolutions=(16, 8),
                attention_num_heads=4,
                channel_mult=(1, 1, 2, 2, 4, 4))
    res = unet.get_intermediates(x, t)

    print("[emb]")
    dump(res["emb"])
    print("")

    print("[down]")
    dump(res["down"])
    print("")

    print("[middle]")
    dump(res["middle"])
    print("")

    print("[up]")
    dump(res["up"])
    print("")

    print("[out]")
    dump(res["out"])
    print("")


def dump(var):
    if isinstance(var, list):
        for x in var:
            dump(x)

        return

    arr = var.d
    mean = arr.mean()
    std = arr.std()
    abs_sum = np.abs(arr).sum()
    print("mean: {:-6.1g} std: {:-6.1g} abs_sum: {:-6.1g} size: {}".format(mean,
          std, abs_sum, arr.size))


@click.command()
@click.option("--loop/--no-loop", default=True)
@click.option("--intermediate/--no-intermediate", default=False)
def test(loop, intermediate):
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
        test_intermediate()


if __name__ == "__main__":
    test()
