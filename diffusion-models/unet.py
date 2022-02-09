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
import numpy as np
import click

import nnabla as nn
import nnabla.functions as F

from layers import *
from utils import Shape4D


class ResidualBlock(object):
    def __init__(self,
                 out_channels=None,
                 scale_shift_norm=False,
                 dropout=0,
                 conv_shortcut=False,
                 channel_last=False):
        self.out_channels = out_channels
        self.scale_shift_norm = scale_shift_norm
        self.dropout = dropout
        self.conv_shortcut = conv_shortcut
        self.channel_last = channel_last

    def in_layers(self, x):
        h = normalize(x, name='norm_in',
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
            h = normalize(
                h, name="norm_out", channel_axis=3 if self.channel_last else 1) * (scale + 1) + shift
        else:
            h += emb
            h = normalize(h, name="norm_out",
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
        else:
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

            # add residual
            out = h + self.shortcut(x)

        return out


class ResidualBlockResampleBase(ResidualBlock):
    def resample(self, x, name):
        raise NotImplementedError("Resample method must be implemented.")

    def in_layers_with_resampling(self, x):
        h = normalize(x, name='norm_in',
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

            # add residual
            out = h + self.shortcut(x_res)

        return out


class ResidualBlockDown(ResidualBlockResampleBase):
    def resample(self, x, name):
        return downsample(x, name, with_conv=False, channel_last=self.channel_last)


class ResidualBlockUp(ResidualBlockResampleBase):
    def resample(self, x, name):
        return upsample(x, name, with_conv=False, channel_last=self.channel_last)


def attn_block(x, name, *, num_heads=4, num_head_channels=None, channel_last=False):
    """Multihead attention block"""
    x_shape = Shape4D(x.shape, channel_last=channel_last)
    B, C, H, W = x_shape.get_as_tuple("bchw")

    if num_head_channels is not None:
        assert isinstance(num_head_channels, int), \
            f"num_head_channels must be an interger but {type(num_head_channels)}"
        assert x_shape.c % num_head_channels == 0, \
            f"input channels (= {C}) is not divisible by num_head_channels (= {num_head_channels})"

        num_heads = C // num_head_channels
    
    assert C % num_heads == 0, \
        f"input channels (= {C}) is not divisible by num_heads (= {num_heads})"
    
    num_head_channels = x_shape.c // num_heads

    with nn.parameter_scope(name):
        # Get query, key, value
        h = normalize(x, name="norm", channel_axis=3 if channel_last else 1)
        qkv = nin(h, 3 * C, name="qkv", channel_last=channel_last)

        if not channel_last:
            # (B, 3 * C, H, W) -> (B, H, W, 3 * C)
            qkv = F.transpose(qkv, (0, 2, 3, 1))
        
        #  always (B, H, W, 3 * C) here
        q, k, v = chunk(qkv, 3, axis=-1)
        
        # scale is applied both q and k to avoid large value in dot op.
        scale = 1 / math.sqrt(math.sqrt(num_head_channels))
        
        q = F.reshape(q * scale, (B * num_heads, H * W, -1))
        k = F.reshape(k * scale, (B * num_heads, H * W, -1))
        v = F.reshape(v, (B * num_heads, H * W, -1))

        # create attention weight
        w = F.batch_matmul(q, k, transpose_b = True) # (B * num_heads, H * W (for q), H * W (for k))
        w = F.softmax(w, axis=-1) # take softmax for each query

        # attention
        a = F.reshape(F.batch_matmul(w, v), (B, H, W, C))

        if not channel_last:
            # (B, H, W, C) -> (B, C, H, W)
            a = F.transpose(a, (0, 3, 1, 2))

        # output projection
        out = nin(a, C, name='proj_out', zeroing_w=True,
                  channel_last=channel_last)

    assert out.shape == x.shape
    return out + x


class UNet(object):
    def __init__(self,
                 num_classes,
                 model_channels,
                 output_channels,
                 num_res_blocks,
                 attention_num_heads,
                 attention_head_channels,
                 attention_resolutions,
                 channel_mult=(1, 2, 4, 8),
                 dropout=0.,
                 scale_shift_norm=False,
                 conv_resample=True,
                 resblock_resample=False,
                 channel_last=False):
        self.num_classes = num_classes
        self.model_channels = model_channels
        self.output_channels = output_channels
        self.num_res_blocks = num_res_blocks
        self.attention_num_heads = attention_num_heads
        self.attention_head_channels = attention_head_channels
        self.attention_resolutions = attention_resolutions
        self.channel_mult = channel_mult
        self.dropout = dropout
        self.scale_shift_norm = scale_shift_norm
        self.conv_resample = conv_resample
        self.resblock_resample = resblock_resample
        self.channel_last = channel_last

    def timestep_embedding(self, t):
        with nn.parameter_scope('timestep_embedding'):
            # sinusoidal embedding
            emb = sinusoidal_embedding(t, self.model_channels)

            # reshape to use conv rather than affine
            if self.channel_last:
                emb = F.reshape(emb, (emb.shape[0], 1, 1, emb.shape[1]))
            else:
                emb = F.reshape(emb, emb.shape + (1, 1))

            # linear transforms
            emb = nin(emb, self.model_channels * 4, name='dense0',
                      channel_last=self.channel_last)
            emb = nonlinearity(emb)

            emb = nin(emb, self.model_channels * 4, name='dense1',
                      channel_last=self.channel_last)

        return emb

    def resblock_with_attention(self, h, emb, out_channels, name):
        with nn.parameter_scope(name):
            block = ResidualBlock(out_channels,
                                  scale_shift_norm=self.scale_shift_norm,
                                  dropout=self.dropout,
                                  channel_last=self.channel_last)

            h = block(h, emb, "res_block")

            res = h.shape[-1]
            if self.attention_resolutions is not None and res in self.attention_resolutions:
                h = attn_block(h, "attention",
                               num_heads=self.attention_num_heads,
                               num_head_channels=self.attention_head_channels,
                               channel_last=self.channel_last)

        return h

    def downsample_blocks(self, h, emb, out_channels, level, down):
        hs = []
        with nn.parameter_scope(f"block_{level}"):
            for i in range(self.num_res_blocks):
                h = self.resblock_with_attention(
                    h, emb, out_channels, name=f"resblock_{i}")
                hs.append(h)

        if down:
            if self.resblock_resample:
                h = ResidualBlockDown(out_channels=out_channels,
                                      scale_shift_norm=self.scale_shift_norm,
                                      dropout=self.dropout,
                                      channel_last=self.channel_last)(h, emb, name=f"downsample_{level}")
            else:
                h = downsample(h,
                               name=f"downsample_{level}",
                               with_conv=self.conv_resample,
                               channel_last=self.channel_last)

            hs.append(h)

        return hs

    def upsample_blocks(self, h, emb, hs_down, out_channels, level, up):
        hs_up = []
        with nn.parameter_scope(f"output_{level}"):
            for i in range(self.num_res_blocks + 1):
                # concat skip
                h = F.concatenate(h, hs_down.pop(),
                                  axis=3 if self.channel_last else 1)
                h = self.resblock_with_attention(
                    h, emb, out_channels, name=f"resblock_{i}")
                hs_up.append(h)

        if up:
            if self.resblock_resample:
                h = ResidualBlockUp(out_channels=out_channels,
                                    scale_shift_norm=self.scale_shift_norm,
                                    dropout=self.dropout,
                                    channel_last=self.channel_last)(h, emb, name=f"upsample_{level}")
            else:
                h = upsample(h,
                             name=f"upsample_{level}",
                             with_conv=self.conv_resample,
                             channel_last=self.channel_last)

            hs_up.pop()  # align with improved-diffusion
            hs_up.append(h)

        return hs_up

    def middle_block(self, h, emb):
        ch = h.shape[-1 if self.channel_last else 1]
        block = ResidualBlock(ch,
                              scale_shift_norm=self.scale_shift_norm,
                              dropout=self.dropout,
                              channel_last=self.channel_last)

        h = block(h, emb, name="resblock_0")

        res = h.shape[1 if self.channel_last else -1]
        if self.attention_resolutions is not None and res in self.attention_resolutions:
            h = attn_block(h, "attention",
                           num_heads=self.attention_num_heads,
                           channel_last=self.channel_last)

        h = block(h, emb, name="resblock_1")

        return h

    def output_block(self, h):
        h = normalize(h, "last_norm",
                      channel_axis=3 if self.channel_last else 1)
        h = nonlinearity(h)
        h = conv(h, self.output_channels, name="last_conv",
                 zeroing_w=True, channel_last=self.channel_last)

        return h

    def get_intermediates(self, x, t, name=None):
        ret = dict()

        with nn.auto_forward(True):
            ch = self.model_channels
            with nn.parameter_scope('UNet' if name is None else name):
                if self.channel_last:
                    # todo: If we allow tf32, might be better to apply pad even if chennel_first.
                    # But, to do that, we have to care obsolete parameters.
                    x = pad_for_faster_conv(x, channel_last=self.channel_last)

                h = conv(x, ch, name="first_conv",
                         channel_last=self.channel_last)
                
                emb = self.timestep_embedding(t)
                ret["emb"] = emb
                emb = nonlinearity(emb)

                hs = [h]
                # downsample block
                with nn.parameter_scope("downsample_block"):
                    for level, mult in enumerate(self.channel_mult):
                        # downsample to lower resolution except last
                        is_last_block = level == len(self.channel_mult) - 1

                        # apply resblock and attention for this resolution
                        outs = self.downsample_blocks(h, emb, ch * mult,
                                                      level=level,
                                                      down=not is_last_block)
                        hs += outs
                        h = outs[-1]

                ret["down"] = hs.copy()

                # middle block
                with nn.parameter_scope("middle_block"):
                    h = self.middle_block(h, emb)

                ret["middle"] = h

                # upsample block
                hs_up = []
                with nn.parameter_scope("upsample_block"):
                    for level, mult in enumerate(reversed(self.channel_mult)):
                        # upsample to larger resolution except last
                        is_last_block = level == len(self.channel_mult) - 1

                        # apply resblock and attention for this resolution
                        outs = self.upsample_blocks(h, emb, hs, ch * mult,
                                                    level=level,
                                                    up=not is_last_block)
                        h = outs[-1]
                        hs_up += outs

                assert len(hs) == 0

                ret["up"] = hs_up.copy()

                # output block
                with nn.parameter_scope("output_block"):
                    out = self.output_block(h)

            ret["out"] = out

            out_shape = Shape4D(x.shape, channel_last=self.channel_last)
            out_shape.c = self.output_channels
            assert Shape4D(
                out.shape, channel_last=self.channel_last) == out_shape

        return ret

    def __call__(self, x, t, name=None):
        ch = self.model_channels
        with nn.parameter_scope('UNet' if name is None else name):
            if self.channel_last:
                # todo: If we allow tf32, might be better to apply pad even if chennel_first.
                # But, to do that, we have to care obsolete parameters.
                x = pad_for_faster_conv(x, channel_last=self.channel_last)

            h = conv(x, ch, name="first_conv", channel_last=self.channel_last)
            emb = self.timestep_embedding(t)
            emb = nonlinearity(emb)

            hs = [h]
            # downsample block
            with nn.parameter_scope("downsample_block"):
                for level, mult in enumerate(self.channel_mult):
                    # downsample to lower resolution except last
                    is_last_block = level == len(self.channel_mult) - 1

                    # apply resblock and attention for this resolution
                    outs = self.downsample_blocks(h, emb, ch * mult,
                                                  level=level,
                                                  down=not is_last_block)
                    hs += outs
                    h = outs[-1]

            # middle block
            with nn.parameter_scope("middle_block"):
                h = self.middle_block(h, emb)

            # upsample block
            hs_up = []
            with nn.parameter_scope("upsample_block"):
                for level, mult in enumerate(reversed(self.channel_mult)):
                    # upsample to larger resolution except last
                    is_last_block = level == len(self.channel_mult) - 1

                    # apply resblock and attention for this resolution
                    outs = self.upsample_blocks(h, emb, hs, ch * mult,
                                                level=level,
                                                up=not is_last_block)
                    h = outs[-1]

            assert len(hs) == 0

            # output block
            with nn.parameter_scope("output_block"):
                out = self.output_block(h)

            out_shape = Shape4D(x.shape, channel_last=self.channel_last)
            out_shape.c = self.output_channels
            assert Shape4D(
                out.shape, channel_last=self.channel_last) == out_shape

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


def test_intermediate(conf=None, h5=None):
    import os

    os.environ["NNABLA_CUDNN_DETERMINISTIC"] = '1'

    nn.clear_parameters()

    if conf is None:
        from neu.misc import AttrDict
        conf = AttrDict({
            "image_shape": (1, 3, 256, 256),
            "base_channels": 128,
            "num_diffusion_timesteps": 1000,
            "num_res_blocks": 3,
            "num_attention_heads": 4,
            "attention_resolutions": (16, 8),
            "channel_mult": (1, 1, 2, 2, 4, 4),
            "ssn": False,
        })
    

    # refinement    
    from diffusion import ModelVarType
    if "model_var_type" in conf:
        conf.model_var_type = ModelVarType.get_vartype_from_key(conf.model_var_type)
    else:
        conf.model_var_type = ModelVarType.FIXED_SMALL
    
    if "channel_last" not in conf:
        conf.channel_last = False
    
    if "num_attention_head_channels" not in conf:
        conf.num_attention_head_channels = None
    
    if "resblock_resample" not in conf:
        conf.resblock_resample = False

    
    if h5 is not None:
        nn.load_parameters(h5)

    np.random.seed(803)
    x = nn.Variable.from_numpy_array(np.random.randn(*((1, ) + conf.image_shape[1:])))
    t = nn.Variable.from_numpy_array(np.random.randint(low=0, high=conf.num_diffusion_timesteps, size=1))
    print(x.d.sum(), t.d)
    
    # x = nn.Variable.from_numpy_array(np.full((1, ) + conf.image_shape[1:], 0.1))
    # t = nn.Variable.from_numpy_array([803])
    
    # define output channel
    from diffusion import is_learn_sigma
    output_channels = x.shape[-1] if conf.channel_last else x.shape[1]
    if is_learn_sigma(conf.model_var_type):
        output_channels *= 2

    unet = UNet(num_classes=1,
                model_channels=conf.base_channels,
                output_channels=output_channels,
                num_res_blocks=conf.num_res_blocks,
                attention_num_heads=conf.num_attention_heads,
                attention_head_channels=conf.num_attention_head_channels,
                attention_resolutions=conf.attention_resolutions,
                channel_mult=conf.channel_mult,
                dropout=0.,
                scale_shift_norm=conf.ssn,
                resblock_resample=conf.resblock_resample,
                channel_last=conf.channel_last)
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
            assert os.path.exists(config), f"config file `{config}` is not found."
            logger.info(f"... with config={config} as followd:")

            from neu.yaml_wrapper import read_yaml
            conf = read_yaml(config)
            conf.dump()

        if h5 is not None:
            assert os.path.exists(h5), f"config file `{h5}` is not found."
            logger.info(f"... with h5={h5}")
        
        test_intermediate(conf, h5)


if __name__ == "__main__":
    test()
