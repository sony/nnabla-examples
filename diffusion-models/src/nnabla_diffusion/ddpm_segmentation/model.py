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


import sys

import nnabla as nn
import nnabla.functions as F
import numpy as np
from nnabla_diffusion.config import (DatasetDDPMConfig, DiffusionConfig,
                                     ModelConfig)
from nnabla_diffusion.diffusion_model.model import Model

from .feature_extractor import FeatureExtractUNet


class FeatureExtractorDDPM(Model):
    def __init__(self,
                 datasetddpm_conf: DatasetDDPMConfig,
                 diffusion_conf: DiffusionConfig,
                 model_conf: ModelConfig):
        super(FeatureExtractorDDPM, self).__init__(diffusion_conf, model_conf)
        self.datasetddpm_conf = datasetddpm_conf
        self.extract_blocks = datasetddpm_conf.blocks
        self.image_shape = datasetddpm_conf.dim[:-1]
        self.upsampling_mode = datasetddpm_conf.upsampling_mode
        self.steps = datasetddpm_conf.steps
        self.ema = datasetddpm_conf.ema

    def _define_model(self):
        net = FeatureExtractUNet(self.extract_blocks, self.model_conf)

        return net

    def extract_features(self, x, model_kwargs=None, noise=None):
        activations = []
        if isinstance(x, np.ndarray):
            x = nn.Variable.from_numpy_array(x)
        if not noise is None:
            noise = nn.Variable.from_numpy_array(noise)
        if model_kwargs is None:
            model_kwargs = {}

        model = self._define_model()
        with nn.auto_forward():
            for t in self.steps:
                T_var = nn.Variable(shape=(1, ))
                T_var.data.fill(t)
                x_t = self.diffusion.q_sample(x, T_var, noise=noise)

                with nn.no_grad():
                    if self.ema:
                        with nn.parameter_scope("ema"):
                            pred, act_t = model(x_t, T_var, **model_kwargs)
                    else:
                        _, act_t = model(x_t, T_var, **model_kwargs)
                activations.extend(act_t)

        return activations

    def collect_features(self, activations):
        resized_activations = []
        with nn.auto_forward():
            for feats in activations:
                feats = F.transpose(feats, (0, 3, 1, 2))
                feats = F.interpolate(feats, output_size=self.image_shape,
                                      mode=self.upsampling_mode)

                resized_activations.append(feats)

            all_activations = F.concatenate(*resized_activations, axis=1)

        return all_activations
