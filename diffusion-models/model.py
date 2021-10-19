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

import nnabla as nn
import nnabla.functions as F
from functools import partial

from diffusion import ModelVarType, is_learn_sigma, get_beta_schedule, const_var, GaussianDiffusion
from unet import UNet

from neu.misc import AttrDict


def respace_betas(betas, use_timesteps):
    """
    Given betas and use_timestpes, returns respaced betas.

    Args:
        betas (np.array): A T elements of array containing beta_t at index t.
        use_timesteps (tuple):
            A tuple indicates which timesteps are used for generation process.
            For example, if use_timestpes = (0, 100, 300, 500, 700, 900, 1000), only ts included in use_timestpes will be used.

    Returns:
        new_betas (np.array): A respaced betas.
        timestep_map (np.array): 
            A list indicating how to map a new index to original index.
            It will be the same as use_timesteps sorted in increasing order.
    """

    if isinstance(use_timesteps, list):
        use_timesteps = tuple(use_timesteps)

    assert isinstance(use_timesteps, tuple)

    T = len(betas)

    new_betas = []
    prev_alphas_cumprod = 1.
    alphas_cumprod = 1.
    timestep_map = []
    for t in range(T):
        alphas_cumprod *= 1. - betas[t]
        if t in use_timesteps:
            new_beta = 1 - alphas_cumprod / prev_alphas_cumprod
            new_betas.append(new_beta)
            timestep_map.append(t)
            prev_alphas_cumprod = alphas_cumprod

    return np.asarray(new_betas, dtype=np.float64), np.asarray(timestep_map, dtype=int)


class Model(object):
    def __init__(self,
                 beta_strategy,
                 num_diffusion_timesteps,
                 use_timesteps=None,
                 num_classes=1,
                 model_var_type: ModelVarType = ModelVarType.FIXED_SMALL,
                 randflip=True,
                 attention_num_heads=1,
                 attention_resolutions=(16, 8),
                 base_channels=None,
                 channel_mult=None,
                 num_res_blocks=None,
                 scale_shift_norm=True):

        betas = get_beta_schedule(
            beta_strategy, num_timesteps=num_diffusion_timesteps)
        self.original_timesteps = len(betas)
        self.timestep_map = None

        if use_timesteps is not None:
            betas, timestep_map = respace_betas(betas, use_timesteps)
            self.timestep_map = const_var(timestep_map)

        self.model_var_type = model_var_type
        self.diffusion: GaussianDiffusion = GaussianDiffusion(betas=betas,
                                                              model_var_type=model_var_type)
        self.num_classes = num_classes
        self.randflip = randflip
        self.attention_num_heads = attention_num_heads
        self.attention_resolutions = attention_resolutions
        self.base_channels = base_channels
        self.channel_mult = channel_mult
        self.num_res_blocks = num_res_blocks
        self.scale_shift_norm = scale_shift_norm

    def _define_model(self, input_shape, dropout):
        assert isinstance(input_shape, (tuple, list))
        assert len(input_shape) == 4

        B, C, H, W = input_shape

        output_channels = C
        if is_learn_sigma(self.model_var_type):
            output_channels *= 2

        unet = UNet(num_classes=self.num_classes,
                    model_channels=self.base_channels,
                    output_channels=output_channels,
                    num_res_blocks=self.num_res_blocks,
                    attention_resolutions=self.attention_resolutions,
                    attention_num_heads=self.attention_num_heads,
                    channel_mult=self.channel_mult,
                    dropout=dropout,
                    scale_shift_norm=self.scale_shift_norm,
                    conv_resample=True)

        return unet

    def rescale_timestep(self, t):
        if self.timestep_map is None:
            return t

        assert isinstance(self.timestep_map, nn.Variable)

        # revert t to the original timestep
        return self.diffusion._extract(self.timestep_map, t)

    def _denoise(self, x, t, dropout):
        assert x.shape[0] == t.shape[0]

        unet = self._define_model(x.shape, dropout)

        out = unet(x, self.rescale_timestep(t))

        if is_learn_sigma(self.model_var_type):
            B, C, H, W = x.shape
            assert out.shape == (B, 2 * C, H, W)
        else:
            assert out.shape == x.shape
        return out

    def get_denoise_net_intermediates(self, x, t, dropout):
        assert x.shape[0] == t.shape[0]

        unet = self._define_model(x.shape, dropout)

        return unet.get_intermediates(x, self.rescale_timestep(t))

    def build_denoise_graph(self, shape):
        """just create parameters for inference."""
        B, _, _, _ = shape
        x_dummy = nn.Variable(shape)
        t_dummy = nn.Variable((B, ))

        out = self._denoise(x_dummy, t_dummy, dropout=0)

        return out

    def build_train_graph(self, x, t=None, dropout=0, noise=None, loss_scaling=None):
        B, C, H, W = x.shape
        if self.randflip:
            x = F.random_flip(x)
            assert x.shape == (B, C, H, W)

        if t is None:
            t = F.randint(
                low=0, high=self.diffusion.num_timesteps, shape=(B, ))
            # F.randint could return high with very low prob. Workaround to avoid this.
            t = F.clip_by_value(t, min=0, max=self.diffusion.num_timesteps-0.5)

        loss_dict = self.diffusion.train_loss(model=partial(self._denoise, dropout=dropout),
                                              x_start=x, t=t, noise=noise)
        assert isinstance(loss_dict, AttrDict)

        # setup training loss
        loss_dict.batched_loss = loss_dict.mse
        if is_learn_sigma(self.model_var_type):
            assert "vlb" in loss_dict
            loss_dict.batched_loss += loss_dict.vlb * 1e-3
            # todo: implement loss aware sampler

        if loss_scaling is not None and loss_scaling > 1:
            loss_dict.batched_loss *= loss_scaling

        # setup flat training loss
        loss_dict.loss = F.mean(loss_dict.batched_loss)
        assert loss_dict.batched_loss.shape == t.shape == (B, )

        # Keep interval values to compute loss for each quantile
        t.persistent = True
        for v in loss_dict.values():
            v.persistent = True

        return loss_dict, t

    def sample(self, shape, dump_interval=-1, noise=None, use_ema=True, progress=False, use_ddim=False):
        if use_ema:
            with nn.parameter_scope("ema"):
                return self.sample(shape,
                                   dump_interval=dump_interval,
                                   noise=noise,
                                   use_ema=False,
                                   progress=progress,
                                   use_ddim=use_ddim)

        loop_func = self.diffusion.ddim_sample_loop if use_ddim else self.diffusion.p_sample_loop

        with nn.no_grad():
            return loop_func(
                model=partial(self._denoise, dropout=0),
                shape=shape,
                noise=noise,
                dump_interval=dump_interval,
                progress=progress
            )

    def sample_trajectory(self, shape, noise=None, use_ema=True, progress=False, use_ddim=False):
        return self.sample(shape,
                           dump_interval=100,
                           noise=noise,
                           use_ema=use_ema,
                           progress=progress,
                           use_ddim=use_ddim)
