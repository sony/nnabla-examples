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

import os
from typing import Optional

import numpy as np
import nnabla as nn

from config import LoadedConfig
from diffusion_model.model import Model


def repeat(v, num):
    return nn.Variable.from_numpy_array([v for _ in range(num)])


class InferenceModel(object):
    def __init__(self,
                 base_conf: LoadedConfig,
                 base_h5: str,
                 up1_conf: Optional[LoadedConfig] = None,
                 up1_h5: Optional[str] = None) -> None:

        # base model
        assert os.path.exists(base_h5), \
            f"h5 file '{base_h5}' for base model doesn't exist."

        self.base_conf = base_conf
        with nn.parameter_scope("base"):
            nn.load_parameters(base_h5)

        # 1st upsampler
        self.use_1st_upsampler = (
            up1_conf is not None) and (up1_h5 is not None)
        if self.use_1st_upsampler:
            assert os.path.exists(up1_h5), \
                f"h5 file '{up1_h5}' for the 1st upsampler model doesn't exist."

            self.up1_conf = up1_conf
            with nn.parameter_scope("up1"):
                nn.load_parameters(up1_h5)

    @staticmethod
    def _infer(conf: LoadedConfig,
               respacing_step,
               sampler,
               class_id,
               classifier_free_guidance_weight,
               *,
               lowres_noise_level=None,
               lowres_image=None):

        num_gen = 8  # todo: pass it from ui
        conf.diffusion.respacing_step = respacing_step

        model = Model(diffusion_conf=conf.diffusion,
                      model_conf=conf.model)

        # set up class condition
        model_kwargs = {}
        if conf.model.class_cond:
            model_kwargs["class_label"] = repeat(int(class_id), num_gen)

        # set up low-res input
        if conf.model.low_res_size is not None:
            assert lowres_image is not None
            lowres_image = nn.Variable.from_numpy_array(lowres_image)

            if conf.model.noisy_low_res:
                assert isinstance(lowres_image, nn.Variable)
                s = repeat(float(lowres_noise_level), num_gen)
                lowres_image, aug_level = model.gaussian_conditioning_augmentation(
                    lowres_image, s)
                model_kwargs["input_cond_aug_timestep"] = aug_level

            model_kwargs["input_cond"] = lowres_image

        # setup sampler
        # todo: refactor to entangle ddim and other sampler methods
        ode_solver = None
        use_ddim = False
        if sampler == "ddim":
            use_ddim = True
        if sampler in ["plms", "dpm2"]:
            ode_solver = sampler

        gen, _, _ = model.sample(shape=(num_gen, ) + conf.model.image_shape,
                                 model_kwargs=model_kwargs,
                                 use_ema=True,
                                 use_ddim=use_ddim,
                                 ode_solver=ode_solver,
                                 progress=True,
                                 classifier_free_guidance_weight=classifier_free_guidance_weight)

        return gen

    def _single_step_inference_callback(self):

        def callback(respacing_step,
                     sampler,
                     class_id,
                     classifier_free_guidance_weight):
            with nn.parameter_scope("base"):
                gen: np.ndarray = self._infer(self.base_conf,
                                              respacing_step,
                                              sampler,
                                              class_id,
                                              classifier_free_guidance_weight)

            return ((gen + 1) * 127.5).astype(np.uint8)

        return callback

    def _two_step_inference_callback(self):

        def callback(base_respacing_step,
                     base_sampler,
                     base_class_id,
                     base_classifier_free_guidance_weight,
                     up_respacing_step,
                     up_sampler,
                     up_class_id,
                     up_classifier_free_guidance_weight,
                     up_lowres_noiose_level
                     ):
            with nn.parameter_scope("base"):
                base_gen: np.ndarray = \
                    self._infer(self.base_conf,
                                base_respacing_step,
                                base_sampler,
                                base_class_id,
                                base_classifier_free_guidance_weight)

            with nn.parameter_scope("up1"):
                gen: np.ndarray = \
                    self._infer(self.up1_conf,
                                up_respacing_step,
                                up_sampler,
                                up_class_id,
                                up_classifier_free_guidance_weight,
                                lowres_noise_level=up_lowres_noiose_level,
                                lowres_image=base_gen)

            return ((gen + 1) * 127.5).astype(np.uint8)

        return callback

    def create_callback(self):
        if not self.use_1st_upsampler:
            return self._single_step_inference_callback()
        else:
            return self._two_step_inference_callback()
