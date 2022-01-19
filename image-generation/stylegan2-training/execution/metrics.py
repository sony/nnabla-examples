# Copyright 2021 Sony Corporation.
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
import sys
from tqdm import tqdm

metrics_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'utils', 'neu'))
sys.path.append(metrics_path)

import nnabla as nn

from .ops import *
from models import *


class Metrics(object):

    def __init__(self, generator, metric_name, n_samples, batch_size):

        self.generator = generator
        self.n_samples = n_samples

        num_batches = n_samples // batch_size
        resid = n_samples - (num_batches * batch_size)
        self.batch_sizes = [batch_size] * num_batches + [resid]
        self.latent_dim = self.generator.mapping_network_dim

        if metric_name == 'ppl':
            from metrics.lpips.lpips import LPIPS
            self.lpips_distance = LPIPS(model='vgg')
            self.eps = 1e-4
        else:
            raise NotImplementedError

    def get_w(self, batch_size, seed=-1):

        z_data = F.randn(shape=(batch_size*2, self.latent_dim)).data
        w_data = mapping_network(z_data)
        w_d_1, w_d_2 = w_data[::2], w_data[1::2]
        latent_t_0 = lerp(w_d_1, w_d_2, 0)
        latent_t_1 = lerp(w_d_1, w_d_2, self.eps)
        latent_t_e = F.stack(latent_t_0, latent_t_1, axis=1)
        latent_t_e = F.reshape(latent_t_e, (w_data.shape))

        return [latent_t_e, latent_t_e]

    def get_fid(self):
        pass

    def get_inception_score(self):
        pass

    def get_ppl(self):

        nn.set_auto_forward(True)

        distances = []
        for bs in tqdm(self.batch_sizes):
            if bs == 0:
                continue

            w = self.get_w(bs)

            # generate output from generator
            constant_bc = nn.parameter.get_parameter_or_create(
                            name="G_synthesis/4x4/Const/const",
                            shape=(1, 512, 4, 4))
            constant_bc = F.broadcast(
                constant_bc, (2*bs,) + constant_bc.shape[1:])
            rgb_output = self.generator.synthesis(w, constant_bc, seed=100)

            # Crop using face prior
            c = rgb_output.shape[2] // 8
            rgb_output = rgb_output[:, :, c * 3: c * 7, c * 2: c * 6]

            factor = rgb_output.shape[2] // 256

            if factor > 1:
                rgb_output = F.reshape(
                    rgb_output, (-1, rgb_output.shape[1], rgb_output.shape[2] // factor, factor, rgb_output.shape[3] // factor, factor))
                rgb_output = F.mean(rgb_output, (3, 5))
            rgb_output_1, rgb_output_2 = rgb_output[::2], rgb_output[1::2]
            dist = self.lpips_distance(nn.Variable.from_numpy_array(
                rgb_output_1.data), nn.Variable.from_numpy_array(rgb_output_2.data)) / (self.eps ** 2)
            distances.append(dist.d.squeeze())

        distances = np.concatenate(distances, 0)

        lo = np.percentile(distances, 1, interpolation="lower")
        hi = np.percentile(distances, 99, interpolation="higher")
        filtered_dist = np.extract(
            np.logical_and(lo <= distances, distances <= hi), distances
        )
        print("PPL:", filtered_dist.mean())
