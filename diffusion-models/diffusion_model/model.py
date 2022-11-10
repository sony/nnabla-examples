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
from neu.misc import AttrDict

from config import DiffusionConfig, ModelConfig

from .diffusion import is_learn_sigma, GaussianDiffusion
from .unet import UNet, EfficientUNet


class Model(object):
    def __init__(self,
                 diffusion_conf: DiffusionConfig,
                 model_conf: ModelConfig):
        self.diffusion = GaussianDiffusion(diffusion_conf)

        # todo: need refactor so as to instantiate model here.
        self.model_conf = model_conf

    def _define_model(self):
        nets = {
            "unet": UNet,
            "efficient_unet": EfficientUNet
        }

        assert self.model_conf.arch in nets, \
            f"model architecture '{self.model_conf.arch}' is not implemented."

        return nets[self.model_conf.arch](self.model_conf)

    def _sampling_timestep(self, n_sample):
        t = F.randint(
                low=0, high=self.diffusion.num_timesteps, shape=(n_sample, ))

        # F.randint could return high with very low prob. Workaround to avoid this.
        t = F.clip_by_value(t, min=0, max=self.diffusion.num_timesteps-0.5)

        t.persistent = True

        return t

    def gaussian_conditioning_augmentation(self, x, s=None):
        """
        Following a defined diffusion noise schedule, 
        add a noise to image `x` corresponding to the given timestep `s`.
        If `s` is None, `s` is randomly sampled from U(0, T-1).

        This is typically used for gaussian conditioning augmentation
        proposed in "Cascaded diffusion models for High Fidelity Image Generation".
        Specifically, returns noisy data x' = q(x_s | x) where s ~ U({0, 1, ..., T-1}) and timestep 's'.
        """
        if s is None:
            B = x.shape[0]
            s = self._sampling_timestep(B)

        return self.diffusion.q_sample(x, s), s

    def build_train_graph(self,
                          x,
                          t=None,
                          noise=None,
                          loss_scaling=None,
                          model_kwargs=None):
        # get input shape before condition
        B = x.shape[0]

        if t is None:
            t = self._sampling_timestep(B)

        loss_dict = self.diffusion.train_loss(model=self._define_model(),
                                              x_start=x,
                                              t=t,
                                              model_kwargs=model_kwargs,
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
               model_kwargs=None,
               use_ema=True,
               dump_interval=-1,
               progress=False,
               use_ddim=False,
               ode_solver=None,
               classifier_free_guidance_weight=None):

        if use_ema:
            with nn.parameter_scope("ema"):
                return self.sample(shape,
                                   noise=noise,
                                   x_start=x_start,
                                   model_kwargs=model_kwargs,
                                   use_ema=False,
                                   dump_interval=dump_interval,
                                   progress=progress,
                                   use_ddim=use_ddim,
                                   ode_solver=ode_solver,
                                   classifier_free_guidance_weight=classifier_free_guidance_weight)

        loop_func = self.diffusion.ddim_sample_loop if use_ddim else self.diffusion.p_sample_loop
        if ode_solver == "plms":
            loop_func = self.diffusion.plms_sample_loop
        elif ode_solver == "dpm2":
            loop_func = self.diffusion.dpm2_sample_loop

        with nn.no_grad():
            return loop_func(
                model=self._define_model(),
                channel_last=self.model_conf.channel_last,
                shape=shape,
                noise=noise,
                x_start=x_start,
                model_kwargs=model_kwargs,
                dump_interval=dump_interval,
                progress=progress,
                classifier_free_guidance_weight=classifier_free_guidance_weight
            )

    def sample_trajectory(self, shape, noise=None, x_start=None, model_kwargs=None, use_ema=True, progress=False, use_ddim=False):
        return self.sample(shape,
                           dump_interval=100,
                           noise=noise,
                           x_start=x_start,
                           model_kwargs=model_kwargs,
                           use_ema=use_ema,
                           progress=progress,
                           use_ddim=use_ddim)
