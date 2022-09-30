# Copyright 2023 Sony Group Corporation.
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
from nnabla_diffusion.config import DiffusionConfig, ModelConfig
from nnabla_diffusion.diffusion_model.layers import *
from nnabla_diffusion.diffusion_model.unet import ResidualBlockUp, UNet
from nnabla_diffusion.diffusion_model.utils import Shape4D, context_scope


class FeatureExtractUNet(UNet):
    def __init__(self,
                 extract_blocks=[5, 6, 7, 8, 12],
                 *args,
                 **kwargs):
        super(FeatureExtractUNet, self).__init__(*args, **kwargs)
        self.extract_blocks = extract_blocks

    def upsample_blocks(self, h, emb, emb_seq, hs_down, out_channels, level, up, resblock_ind, num_res_block):
        with nn.parameter_scope(f"output_{level}"):
            # concat skip
            skip = hs_down.pop()
            h = F.concatenate(h, skip,
                              axis=3 if self.conf.channel_last else 1)
            h = self.resblock_with_attention(
                h, emb, emb_seq, out_channels, name=f"resblock_{resblock_ind}")

        if up and resblock_ind == num_res_block:
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

    def __call__(self, x, t, *, name=None, **model_kwargs):
        extract_output = []
        ch = self.conf.base_channels
        with context_scope("half" if self.use_mixed_precision else "float"), nn.parameter_scope('UNet' if name is None else name):
            # condition
            x, emb, emb_seq = self.condition_processing(x, t, **model_kwargs)

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

                    for i in range(num_res_block + 1):
                        h = self.upsample_blocks(h,
                                                 emb,
                                                 emb_seq,
                                                 hs,
                                                 ch * mult,
                                                 level=level,
                                                 up=not is_last_block,
                                                 resblock_ind=i,
                                                 num_res_block=num_res_block)
                        if level == 0:
                            layer_counter = i
                        else:
                            layer_counter = (level *
                                             (self.num_res_blocks[level - 1] + 1) + (i))
                        if layer_counter in self.extract_blocks:
                            extract_output.append(h)

            assert len(hs) == 0

            # output block
            with nn.parameter_scope("output_block"):
                out = self.output_block(h)

            out_shape = Shape4D(x.shape, channel_last=self.conf.channel_last)
            out_shape.c = self.conf.output_channels
            assert Shape4D(out.shape,
                           channel_last=self.conf.channel_last) == out_shape

        return out, extract_output
