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

import os
import sys
import argparse
import numpy as np
from tqdm import trange
from PIL import Image

import nnabla as nn
import nnabla.functions as F
from nnabla.logger import logger
from nnabla.utils.image_utils import imsave

pix2pixhd_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'pix2pixHD'))
sys.path.append(pix2pixhd_path)
from models import LocalGenerator
from utils import *


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-path", "-L", required=True, type=str)
    parser.add_argument("--test-input-dir", "-ti", required=True, type=str)
    parser.add_argument("--test-output-dir", "-to",
                        default='results/test', type=str)
    # parser.add_argument("--num-test-samples", "-N", default=-1, type=int)
    args, subargs = parser.parse_known_args()

    config = read_yaml(os.path.join('configs', 'gender.yaml'))

    config.load_path = args.load_path
    config.test_input_dir = args.test_input_dir
    config.test_output_dir = args.test_output_dir
    # config.num_test_samples = args.num_test_samples

    return config


def get_var(path, image_shape):
    image = Image.open(path).convert("RGB").resize(
        image_shape, resample=Image.BILINEAR)
    image = np.array(image)/255.0
    image = np.transpose(image.astype(np.float32), (2, 0, 1))
    image = np.expand_dims(image, 0)

    image = (image - 0.5)/(0.5)
    return image


def generate():
    config = get_config()

    # batch_size is forced to be 1
    config.train.batch_size = 1

    image_shape = (config.train.batch_size, 3) + \
        tuple(x * config.model.g_n_scales for x in [512, 512])

    # set context
    comm = init_nnabla(config.nnabla_context)

    img_path_list = [os.path.join(config.test_input_dir, path)
                     for path in os.listdir(config.test_input_dir)]

    test_image = nn.Variable(shape=image_shape)
    # define generator
    generator = LocalGenerator()
    generated_image, _, = generator(test_image,
                                    lg_channels=config.model.lg_channels,
                                    gg_channels=config.model.gg_channels,
                                    n_scales=config.model.g_n_scales,
                                    lg_n_residual_layers=config.model.lg_num_residual_loop,
                                    gg_n_residual_layers=config.model.gg_num_residual_loop)

    # load parameters
    if not os.path.exists(config.load_path):
        logger.warn("Path to load params is not found."
                    " Loading params is skipped and generated result will be unreasonable. ({})".format(config.load_path))

    nn.load_parameters(config.load_path)

    progress_iterator = trange(len(img_path_list) // config.train.batch_size,
                               desc="[Generating Images]", disable=comm.rank > 0)

    save_path = os.path.join(config.test_output_dir)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    for i in progress_iterator:
        path = img_path_list[i]
        test_image_data = get_var(path, image_shape[2:])
        test_image.d = test_image_data

        generated_image.forward(clear_buffer=True)

        generated_image_data = (generated_image.d - generated_image.d.min()) / \
            (generated_image.d.max() - generated_image.d.min())
        test_image_data = test_image_data*0.5+0.5

        gen_image_path = os.path.join(
            save_path, "res{}_{}.png".format(comm.rank, i))
        input_image_path = os.path.join(
            save_path, "input_{}_{}.png".format(comm.rank, i))

        imsave(gen_image_path, generated_image_data[0], channel_first=True)
        imsave(input_image_path, test_image_data[0], channel_first=True)


if __name__ == '__main__':
    generate()
