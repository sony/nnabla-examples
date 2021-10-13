# Copyright 2020,2021 Sony Corporation.
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
import os
import argparse
from scipy.stats import truncnorm
import cv2

import nnabla as nn
from nnabla.ext_utils import get_extension_context

from generator import Generator


def sample():

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", "-b", type=int, default=8)
    parser.add_argument("--model", "-m", type=str, default='')
    parser.add_argument("--image-size", "-i", type=int, default=256)
    parser.add_argument("--num", "-n", type=int, default=5)
    parser.add_argument("--trunc", "-t", type=int, default=-1)
    parser.add_argument("--trunc-sigma", "-ts", type=float, default=1.0)
    parser.add_argument("--out", "-o", type=str, default='out.png')
    args = parser.parse_args()

    batch_size = args.batch_size
    latent = 256
    image_size = args.image_size

    ctx = get_extension_context("cudnn", device_id=0)
    nn.set_default_context(ctx)

    scope_gen = "Generator"
    z = nn.Variable([batch_size, latent, 1, 1])
    x_fake = Generator(z, scope_name=scope_gen,
                       img_size=image_size, train=True)[0]
    with nn.parameter_scope(scope_gen):
        nn.load_parameters(args.model)

    x_save = np.zeros((image_size * batch_size, image_size *
                       (args.num+2), 3), dtype=np.float32)
    if args.trunc > 0:
        z0 = args.trunc_sigma * \
            truncnorm.rvs(-args.trunc_sigma, args.trunc_sigma,
                          size=(batch_size, latent, 1, 1))
        z1 = args.trunc_sigma * \
            truncnorm.rvs(-args.trunc_sigma, args.trunc_sigma,
                          size=(batch_size, latent, 1, 1))
    else:
        z0 = np.random.randn(batch_size, latent, 1, 1)
        z1 = np.random.randn(batch_size, latent, 1, 1)

    z.d = z0
    x_fake.forward(clear_buffer=True)
    for b in range(batch_size):
        x_save[b*image_size:(b+1)*image_size,
               0:image_size] = x_fake.d[b].transpose((1, 2, 0))[:, :, ::-1]
    z.d = z1
    x_fake.forward(clear_buffer=True)
    for b in range(batch_size):
        x_save[b*image_size:(b+1)*image_size, -
               image_size:] = x_fake.d[b].transpose((1, 2, 0))[:, :, ::-1]

    for i in range(args.num):
        alpha = (i+1) / (args.num+1)
        z.d = alpha * z1 + (1.0 - alpha) * z0
        x_fake.forward(clear_buffer=True)
        for b in range(batch_size):
            x_save[b*image_size:(b+1)*image_size, (i+1)*image_size:(i+2)
                   * image_size] = x_fake.d[b].transpose((1, 2, 0))[:, :, ::-1]

    cv2.imwrite(args.out, (x_save+1)*128)


if __name__ == '__main__':
    sample()
