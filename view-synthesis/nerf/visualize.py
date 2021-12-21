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

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
from nnabla.ext_utils import get_extension_context

import numpy as np
import imageio
from tqdm import tqdm, trange
import argparse
import os
import sys

from train.nerf import forward_pass
from train.common import *
from data_iterator.get_data import get_data


common_utils_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))
sys.path.append(common_utils_path)
from neu.yaml_wrapper import read_yaml, write_yaml


def trans_t(t): return np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1],
], dtype=np.float32)


def rot_phi(phi): return np.array([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1],
], dtype=np.float32)


def rot_theta(th): return np.array([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1],
], dtype=np.float32)


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0],
                    [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
    return c2w


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--output-filename', '-o', type=str, default='video.gif',
                        help="name of an output file.")
    parser.add_argument('--output-static-filename', '-os', type=str, default='video_static.gif',
                        help="name of an output file.")
    parser.add_argument('--config-path', '-c', type=str, default='configs/llff.yaml',
                        required=True,
                        help='model and training configuration file')
    parser.add_argument('--weight-path', '-w', type=str, default='configs/llff.yaml',
                        required=True,
                        help='path to pretrained NeRF parameters')
    parser.add_argument('--model', type=str, choices=['wild', 'uncertainty', 'appearance', 'vanilla'],
                        required=True,
                        help='Select the model to train')

    parser.add_argument('--visualization-type', '-v', type=str,
                        choices=['zoom', '360-rotation', 'default'],
                        default='default-render-poses',
                        help='type of visualization')

    parser.add_argument('--downscale', '-d', default=1, type=float,
                        help="downsampling factor for the rendered images for faster inference")

    parser.add_argument('--num-images', '-n', default=120, type=int,
                        help="Number of images to generate for the output video/gif")

    parser.add_argument("--fast", help="Use Fast NeRF architecture",
                        action="store_true")

    args = parser.parse_args()

    use_transient = False
    use_embedding = False

    if args.model == 'wild':
        use_transient = True
        use_embedding = True
    elif args.model == 'uncertainty':
        use_transient = True
    elif args.model == 'appearance':
        use_embedding = True

    args = parser.parse_args()
    config = read_yaml(args.config_path)

    config.data.downscale = args.downscale

    nn.set_auto_forward(True)
    ctx = get_extension_context('cuda')
    nn.set_default_context(ctx)
    nn.load_parameters(args.weight_path)

    _, _, render_poses, hwf, _, _, near_plane, far_plane = get_data(config)
    height, width, focal_length = hwf
    print(
        f'Rendering with Height {height}, Width {width}, Focal Length: {focal_length}')

    # mapping_net = MLP
    encode_position_function = get_encoding_function(
        config.train.num_encodings_position, True, True)
    if config.train.use_view_directions:
        encode_direction_function = get_encoding_function(
            config.train.num_encodings_direction, True, True)
    else:
        encode_direction_function = None

    frames = []
    if use_transient:
        static_frames = []

    if args.visualization_type == '360-rotation':
        print('The 360 degree roation result will not work with LLFF data!')
        pbar = tqdm(np.linspace(0, 360, args.num_images, endpoint=False))
    elif args.visualization_type == 'zoom':
        pbar = tqdm(np.linspace(near_plane, far_plane,
                                args.num_images, endpoint=False))
    else:
        args.num_images = min(args.num_images, render_poses.shape[0])
        pbar = tqdm(
            np.arange(0, render_poses.shape[0], render_poses.shape[0]//args.num_images))

    print(f'Rendering {args.num_images} poses...')

    for th in pbar:

        if args.visualization_type == '360-rotation':
            pose = nn.NdArray.from_numpy_array(pose_spherical(th, -30., 4.))
        elif args.visualization_type == 'zoom':
            pose = nn.NdArray.from_numpy_array(trans_t(th))
        else:
            pose = nn.NdArray.from_numpy_array(render_poses[th][:3, :4])
            # pose = nn.NdArray.from_numpy_array(render_poses[0][:3, :4])

        ray_directions, ray_origins = get_ray_bundle(
            height, width, focal_length, pose)

        ray_directions = F.reshape(ray_directions, (-1, 3))
        ray_origins = F.reshape(ray_origins, (-1, 3))

        num_ray_batches = ray_directions.shape[0]//config.train.ray_batch_size+1

        app_emb, trans_emb = None, None
        if use_embedding:
            with nn.parameter_scope('embedding_a'):
                embed_inp = nn.NdArray.from_numpy_array(
                    np.full((config.train.chunksize_fine,), 1, dtype=int))
                app_emb = PF.embed(
                    embed_inp, config.train.n_vocab, config.train.n_app)

        if use_transient:
            with nn.parameter_scope('embedding_t'):
                embed_inp = nn.NdArray.from_numpy_array(
                    np.full((config.train.chunksize_fine,), th, dtype=int))
                trans_emb = PF.embed(
                    embed_inp, config.train.n_vocab, config.train.n_trans)

            static_rgb_map_fine_list, transient_rgb_map_fine_list = [], []

        rgb_map_fine_list = []

        for i in trange(num_ray_batches):
            if i != num_ray_batches-1:
                ray_d, ray_o = ray_directions[i*config.train.ray_batch_size:(
                    i+1)*config.train.ray_batch_size], ray_origins[i*config.train.ray_batch_size:(i+1)*config.train.ray_batch_size]
            else:
                ray_d, ray_o = ray_directions[i*config.train.ray_batch_size:,
                                              :], ray_origins[i*config.train.ray_batch_size:, :]

            if use_transient:
                _, rgb_map_fine, static_rgb_map_fine, transient_rgb_map_fine, _, _, _ = forward_pass(ray_d, ray_o, near_plane, far_plane,
                                                                                                     app_emb, trans_emb, encode_position_function, encode_direction_function, config, use_transient, hwf=hwf, fast=args.fast)

                static_rgb_map_fine_list.append(static_rgb_map_fine)
                transient_rgb_map_fine_list.append(transient_rgb_map_fine)

            else:
                _, _, _, _, rgb_map_fine, _, _, _ = \
                    forward_pass(ray_d, ray_o, near_plane, far_plane, app_emb, trans_emb, encode_position_function,
                                 encode_direction_function, config, use_transient, hwf=hwf, fast=args.fast)
            rgb_map_fine_list.append(rgb_map_fine)

        rgb_map_fine = F.concatenate(*rgb_map_fine_list, axis=0)
        rgb_map_fine = F.reshape(rgb_map_fine, (height, width, 3))

        if use_transient:
            static_rgb_map_fine = F.concatenate(
                *static_rgb_map_fine_list, axis=0)
            static_rgb_map_fine = F.reshape(
                static_rgb_map_fine, (height, width, 3))

        frames.append((255*np.clip(rgb_map_fine.data, 0, 1)).astype(np.uint8))
        if use_transient:
            static_frames.append(
                (255*np.clip(static_rgb_map_fine.data, 0, 1)).astype(np.uint8))

    imageio.mimwrite(args.output_filename, frames, fps=30)
    if use_transient:
        imageio.mimwrite(args.output_static_filename, static_frames, fps=30)


if __name__ == '__main__':
    main()
