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

import numpy as np
import os

from .base import BaseExecution
from .ops import *
from models import *


class Evaluate(BaseExecution):

    def __init__(self, monitor, config, args, comm, few_shot_config):
        super(Evaluate, self).__init__(
            monitor, config, args, comm, few_shot_config)

        for test in args.test:
            eval(f'self.{test}(config["test"], args)')

    def generate(self, test_config, args):
        """Generator Inference
        Args:
                test_config (yaml file obj): Testing configuration such as seed values, mixing configuration
        """

        assert 0 < test_config['mix_after'] < self.generator.num_conv_layers - \
            1, f'specify mix_after from 1 to {self.generator.num_conv_layers-2}'

        print('Testing generation...')
        print(f'truncation value: {test_config["truncation_psi"]}')
        print(f'seed for additional noise: {test_config["stochastic_seed"]}')

        if test_config['mixing']:
            print(
                f'using style noise seed {test_config["mix_seed"][0]} for layers 0-{test_config["mix_after"] - 1}')
            print(
                f'using style noise seed {test_config["mix_seed"][1]} for layers {test_config["mix_after"]}-{self.generator.num_conv_layers}.')
        else:
            print(
                f'using style noise seed {test_config["mix_seed"][0]} for entire layers.')

        style_noises_data = mixing_noise(
            self.batch_size, test_config['latent_dim'], 1, seed=test_config['mix_seed'])

        style_noises = [nn.NdArray.from_numpy_array(style_noises_data[0])]
        style_noises += [nn.NdArray.from_numpy_array(style_noises_data[1])]

        rgb_output = self.generator(
            self.batch_size, style_noises, test_config['truncation_psi'])

        if not self.auto_forward and isinstance(rgb_output, nn.Variable):
            rgb_output.forward(clear_buffer=True)

        filename = os.path.join(self.results_dir, test_config["output_name"])

        save_generations(rgb_output, filename)

    def latent_space_interpolation(self, test_config, args):
        from PIL import Image
        assert 0 < test_config['mix_after'] < self.generator.num_conv_layers - \
            1, f'specify mix_after from 1 to {self.generator.num_conv_layers-2}'

        print('Testing interpolation of generation...')

        noise_data_1 = mixing_noise(
            self.batch_size, test_config['latent_dim'], 1, seed=args.seed_1)
        noise_data_2 = mixing_noise(
            self.batch_size, test_config['latent_dim'], 1, seed=args.seed_2)

        style_noise_1 = [nn.NdArray.from_numpy_array(
            noise_data_1[0]), nn.NdArray.from_numpy_array(noise_data_1[1])]
        style_noise_2 = [nn.NdArray.from_numpy_array(
            noise_data_2[0]), nn.NdArray.from_numpy_array(noise_data_2[1])]

        ratios = np.linspace(0, 4, 10)

        images = []
        for r in ratios:
            w = slerp(style_noise_1, style_noise_2, r)
            rgb_output = self.generator(
                self.batch_size, w, test_config['truncation_psi'])
            image = save_generations(rgb_output, None, return_images=True)
            image = np.concatenate([img for img in image], axis=2)
            images.append(Image.fromarray(image.transpose(1, 2, 0)))

        filename = f'{self.results_dir}/{str(args.seed_1)}_{str(args.seed_2)}.gif'
        images[0].save(filename, save_all=True, append_images=images[1:],
                       duration=80, loop=0, optimize=True)

        for i in range(len(images)):
            filename = f'{self.results_dir}/frame_{i}.png'
            images[i].save(filename)

        print(f'Interpolation resuls saved in {self.results_dir}!')

    def style_mixing(self, test_config, args):

        from nnabla.utils.image_utils import imsave, imresize

        print('Testing style mixing of generation...')

        z1 = F.randn(shape=(args.batch_size_A,
                     test_config['latent_dim']), seed=args.seed_1[0]).data
        z2 = F.randn(shape=(args.batch_size_B,
                     test_config['latent_dim']), seed=args.seed_2[0]).data

        nn.set_auto_forward(True)

        mix_image_stacks = []
        for i in range(args.batch_size_A):
            image_column = []
            for j in range(args.batch_size_B):
                style_noises = [
                    F.reshape(z1[i], (1, 512)), F.reshape(z2[j], (1, 512))]
                rgb_output = self.generator(
                    1, style_noises, test_config['truncation_psi'], mixing_layer_index=test_config['mix_after'])
                image = save_generations(rgb_output, None, return_images=True)
                image_column.append(image[0])
            image_column = np.concatenate(
                [image for image in image_column], axis=1)
            mix_image_stacks.append(image_column)
        mix_image_stacks = np.concatenate(
            [image for image in mix_image_stacks], axis=2)

        style_noises = [z1, z1]
        rgb_output = self.generator(
            args.batch_size_A, style_noises, test_config['truncation_psi'])
        image_A = save_generations(rgb_output, None, return_images=True)
        image_A = np.concatenate([image for image in image_A], axis=2)

        style_noises = [z2, z2]
        rgb_output = self.generator(
            args.batch_size_B, style_noises, test_config['truncation_psi'])
        image_B = save_generations(rgb_output, None, return_images=True)
        image_B = np.concatenate([image for image in image_B], axis=1)

        top_image = 255 * np.ones(rgb_output[0].shape).astype(np.uint8)

        top_image = np.concatenate((top_image, image_A), axis=2)
        grid_image = np.concatenate((image_B, mix_image_stacks), axis=2)
        grid_image = np.concatenate((top_image, grid_image), axis=1)

        filename = os.path.join(self.results_dir, 'style_mix.png')
        imsave(filename, imresize(grid_image, (1024, 1024),
               channel_first=True), channel_first=True)
        print(f'Output saved as {filename}')

    def latent_space_projection(self, test_config, args):

        from .projection import LatentSpaceProjection

        lsp = LatentSpaceProjection(self.generator, args)

    def ppl(self, test_config, args):

        from .metrics import Metrics

        metric = Metrics(self.generator, 'ppl', 5000, args.batch_size)
        metric.get_ppl()
