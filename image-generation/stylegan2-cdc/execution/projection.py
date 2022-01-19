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


import nnabla as nn
import nnabla.functions as F
import nnabla.solvers as S

import numpy as np
import os
import sys
from PIL import Image
import subprocess as sp
from tqdm import tqdm

from .ops import *
from models import *

metrics_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', '..', 'utils', 'neu'))
sys.path.append(metrics_path)
from metrics.lpips.lpips import LPIPS


class LatentSpaceProjection(object):

    def __init__(self, generator, args):

        self.generator = generator

        self.solver = S.Adam()
        self.base_lr = 0.1

        self.img_size = 1024
        self.n_latent = 10000
        self.num_iters = 500
        self.latent_dim = self.generator.mapping_network_dim
        self.mse_c = 0.0
        self.n_c = 1e5

        self.lpips_distance = LPIPS(model='vgg')

        self.project(args)

    def set_lr(self, t, rampdown=0.25, rampup=0.05):
        lr_ramp = min(1, (1 - t) / rampdown)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1, t / rampup)

        self.solver.set_learning_rate(self.base_lr * lr_ramp)

    def latent_noise(self, latent, strength):
        noise = F.randn(shape=latent.shape)*strength.data
        return noise + latent

    def regularize_noise(self, noises):
        loss = 0
        for noise in noises:
            size = noise.shape[2]
            while True:
                loss = (loss
                        + F.pow_scalar(F.mean(noise * F.shift(noise,
                                       shifts=(0, 0, 0, 1), border_mode='reflect')), 2)
                        + F.pow_scalar(F.mean(noise * F.shift(noise, shifts=(0, 0, 1, 0), border_mode='reflect')), 2))
                if size <= 8:
                    break
                noise = F.reshape(noise, [-1, 1, size // 2, 2, size // 2, 2])
                noise = F.mean(noise, [3, 5])
                size //= 2
        return loss

    def normalize_noises(self, noises):
        for i in range(len(noises)):
            mean = np.mean(noises[i].d, keepdims=True)
            std = np.std(noises[i].d, keepdims=True)

            noises[i].d = (noises[i].d-mean)/std
        return noises

    def project(self, args):
        nn.set_auto_forward(True)

        # Input Image Variable
        image = Image.open(args.img_path).convert(
            "RGB").resize((256, 256), resample=Image.BILINEAR)
        image = np.array(image)/255.0
        image = np.transpose(image.astype(np.float32), (2, 0, 1))
        image = np.expand_dims(image, 0)
        image = (image - 0.5)/(0.5)
        image = nn.Variable.from_numpy_array(image)

        # Get Latent Space Mean and Std. Dev.
        # Get Noise for B network
        z = F.randn(shape=(self.n_latent, self.latent_dim)).data
        w = mapping_network(z)
        latent_mean = F.mean(w, axis=0, keepdims=True)
        latent_std = F.pow_scalar(F.mean(F.pow_scalar(w-latent_mean, 2)), 0.5)

        # Get Noise
        noises = [F.randn(shape=(1, 1, 4, 4)).data]

        for res in self.generator.resolutions[1:]:
            for _ in range(2):
                shape = (1, 1, res, res)
                noises.append(F.randn(shape=shape).data)

        # Prepare parameters to be optimized
        latent_in = nn.Variable.from_numpy_array(
            latent_mean.data).apply(need_grad=True)
        noises = [nn.Variable.from_numpy_array(
            n.data).apply(need_grad=True) for n in noises]

        constant_bc = nn.parameter.get_parameter_or_create(
                        name="G_synthesis/4x4/Const/const",
                        shape=(1, 512, 4, 4))
        constant_bc = F.broadcast(constant_bc, (1,) + constant_bc.shape[1:])

        pbar = tqdm(range(self.num_iters))
        for i in pbar:

            t = i/self.num_iters
            self.set_lr(t)

            noise_strength = latent_std * 0.05 * max(0, 1 - t / 0.75) ** 2
            latent_n = self.latent_noise(latent_in, noise_strength)

            gen_out = self.generator.synthesis(
                [latent_n, latent_n], constant_bc, noises_in=noises)
            N, C, H, W = gen_out.shape
            factor = H//256
            gen_out = F.reshape(
                gen_out, (N, C, H//factor, factor, W//factor, factor))
            gen_out = F.mean(gen_out, axis=(3, 5))

            p_loss = F.sum(self.lpips_distance(image, gen_out))
            n_loss = self.regularize_noise(noises)
            mse_loss = F.mean((gen_out-image)**2)
            loss = p_loss + self.n_c*n_loss + self.mse_c*mse_loss

            param_dict = {'latent': latent_in}
            for i in range(len(noises)):
                param_dict[f'noise_{i}'] = noises[i]
            self.solver.zero_grad()
            self.solver.set_parameters(
                param_dict, reset=False, retain_state=True)

            loss.backward()
            self.solver.update()

            noises = self.normalize_noises(noises)

            pbar.set_description(f'Loss: {loss.d} P Loss: {p_loss.d}')

        save_generations(image, 'original.png')

        gen_out = self.generator.synthesis(
            [latent_n, latent_n], constant_bc, noises_in=noises)
        N, C, H, W = gen_out.shape
        factor = H//256
        gen_out = F.reshape(
            gen_out, (N, C, H//factor, factor, W//factor, factor), inplace=True)
        gen_out = F.mean(gen_out, axis=(3, 5))
        save_generations(gen_out, 'projected.png')

        nn.save_parameters('projection_params.h5', param_dict)
