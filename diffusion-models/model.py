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
from functools import partial

from diffusion import is_learn_sigma, GaussianDiffusion
from unet import UNet
from config import DiffusionConfig, ModelConfig

from neu.misc import AttrDict

class Model(object):
    def __init__(self,
                 diffusion_conf: DiffusionConfig,
                 model_conf: ModelConfig):
        self.diffusion = GaussianDiffusion(diffusion_conf)

        # todo: need refactor so as to instantiate model here.
        self.model_conf = model_conf  

    def _define_model(self):
        unet = UNet(self.model_conf)
        return unet

    def build_train_graph(self, x, t=None, noise=None, loss_scaling=None):
        # get input shape before condition
        B = x.shape[0]

        if t is None:
            t = F.randint(
                low=0, high=self.diffusion.num_timesteps, shape=(B, ))
            # F.randint could return high with very low prob. Workaround to avoid this.
            t = F.clip_by_value(t, min=0, max=self.diffusion.num_timesteps-0.5)

        loss_dict = self.diffusion.train_loss(model=self._define_model(),
                                              x_start=x,
                                              t=t,
                                              noise=noise,
                                              channel_last=self.model_conf.channel_last)
        assert isinstance(loss_dict, AttrDict)

        # setup training loss
        loss_dict.batched_loss = loss_dict.mse
        if is_learn_sigma(self.model_conf.model_var_type):
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

    def sample(self,
               shape, 
               *,
               noise=None,
               x_start=None,
               use_ema=True, 
               dump_interval=-1, 
               progress=False, 
               use_ddim=False):
        
        if use_ema:
            with nn.parameter_scope("ema"):
                return self.sample(shape,
                                   dump_interval=dump_interval,
                                   noise=noise,
                                   x_start=x_start,
                                   use_ema=False,
                                   dump_interval=dump_interval,
                                   progress=progress,
                                   use_ddim=use_ddim)

        loop_func = self.diffusion.ddim_sample_loop if use_ddim else self.diffusion.p_sample_loop

        with nn.no_grad():
            return loop_func(
                model=self._define_model(),
                channel_last=self.model_conf.channel_last,
                shape=shape,
                noise=noise,
                x_start=x_start,
                dump_interval=dump_interval,
                progress=progress
            )

    def sample_trajectory(self, shape, noise=None, x_start=None, use_ema=True, progress=False, use_ddim=False):
        return self.sample(shape,
                           dump_interval=100,
                           noise=noise,
                           x_start=x_start,
                           use_ema=use_ema,
                           progress=progress,
                           use_ddim=use_ddim)
