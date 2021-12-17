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
from nnabla.utils.data_iterator import data_iterator


import numpy as np
import imageio
from tqdm import tqdm, trange
import argparse
import os
import sys

from train.nerf import forward_pass
from train.common import *
from data_iterator import get_photo_tourism_dataiterator


common_utils_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))
sys.path.append(common_utils_path)
from neu.yaml_wrapper import read_yaml
from neu.misc import init_nnabla


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

    args = parser.parse_args()

    nn.set_auto_forward(True)
    comm = init_nnabla(ext_name="cudnn")

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
    nn.load_parameters(args.weight_path)

    data_source = get_photo_tourism_dataiterator(config, 'test', comm)

    # Pose, Appearance index for generating novel views
    # as well as camera trajectory is hard-coded here.
    data_source.test_appearance_idx = 125
    pose_idx = 125
    dx = np.linspace(-0.2, 0.15, args.num_images//3)
    dy = -0.15
    dz = np.linspace(0.1, 0.22, args.num_images//3)

    embed_idx_list = list(data_source.poses_dict.keys())

    data_source.poses_test = np.tile(
        data_source.poses_dict[pose_idx], (args.num_images, 1, 1))
    for i in range(0, args.num_images//3):
        data_source.poses_test[i, 0, 3] += dx[i]
        data_source.poses_test[i, 1, 3] += dy
    for i in range(args.num_images//3, args.num_images//2):
        data_source.poses_test[i, 0, 3] += dx[len(dx)-1-i]
        data_source.poses_test[i, 1, 3] += dy

    for i in range(args.num_images//2, 5*args.num_images//6):
        data_source.poses_test[i, 2, 3] += dz[i-args.num_images//2]
        data_source.poses_test[i, 1, 3] += dy
        data_source.poses_test[i, 0, 3] += dx[len(dx)//2]

    for i in range(5*args.num_images//6, args.num_images):
        data_source.poses_test[i, 2, 3] += dz[args.num_images-1 - i]
        data_source.poses_test[i, 1, 3] += dy
        data_source.poses_test[i, 0, 3] += dx[len(dx)//2]

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

    pbar = tqdm(np.arange(0, data_source.poses_test.shape[0]))
    data_source._size = data_source.poses_test.shape[0]
    data_source.test_img_w = 400
    data_source.test_img_h = 400
    data_source.test_focal = data_source.test_img_w/2/np.tan(np.pi/6)
    data_source.test_K = np.array([[data_source.test_focal, 0, data_source.test_img_w/2],
                                   [0, data_source.test_focal,
                                       data_source.test_img_h/2],
                                   [0, 0, 1]])

    data_source._indexes = np.arange(0, data_source._size)

    di = data_iterator(data_source, batch_size=1)

    print(f'Rendering {args.num_images} poses...')

    a = [1, 128]
    alpha = np.linspace(0, 1, args.num_images)

    for th in pbar:

        rays, embed_inp = di.next()
        ray_origins = nn.NdArray.from_numpy_array(rays[0, :, :3])
        ray_directions = nn.NdArray.from_numpy_array(rays[0, :, 3:6])
        near_plane_ = nn.NdArray.from_numpy_array(rays[0, :, 6])
        far_plane_ = nn.NdArray.from_numpy_array(rays[0, :, 7])

        embed_inp = nn.NdArray.from_numpy_array(
            embed_inp[0, :config.train.chunksize_fine])
        image_shape = (data_source.test_img_w, data_source.test_img_h, 3)

        ray_directions = F.reshape(ray_directions, (-1, 3))
        ray_origins = F.reshape(ray_origins, (-1, 3))

        num_ray_batches = (
            ray_directions.shape[0] + config.train.ray_batch_size-1)//config.train.ray_batch_size

        app_emb, trans_emb = None, None
        if use_embedding:
            with nn.parameter_scope('embedding_a'):
                embed_inp_app = nn.NdArray.from_numpy_array(
                    np.full((config.train.chunksize_fine,), a[0], dtype=int))
                app_emb = PF.embed(
                    embed_inp_app, config.train.n_vocab, config.train.n_app)

                embed_inp_app = nn.NdArray.from_numpy_array(
                    np.full((config.train.chunksize_fine,), a[1], dtype=int))
                app_emb_2 = PF.embed(
                    embed_inp_app, config.train.n_vocab, config.train.n_app)

                app_emb = app_emb * alpha[th] + app_emb_2 * (1-alpha[th])

        if use_transient:
            with nn.parameter_scope('embedding_t'):
                trans_emb = PF.embed(
                    embed_inp, config.train.n_vocab, config.train.n_trans)

            static_rgb_map_fine_list, transient_rgb_map_fine_list = [], []

        rgb_map_fine_list = []

        for i in trange(num_ray_batches):
            ray_d, ray_o = ray_directions[i*config.train.ray_batch_size:(
                i+1)*config.train.ray_batch_size], ray_origins[i*config.train.ray_batch_size:(i+1)*config.train.ray_batch_size]

            near_plane = near_plane_[i*config.train.ray_batch_size:(
                            i+1)*config.train.ray_batch_size]
            far_plane = far_plane_[i*config.train.ray_batch_size:(
                i+1)*config.train.ray_batch_size]

            if use_transient:
                _, rgb_map_fine, static_rgb_map_fine, transient_rgb_map_fine, _, _, _ = forward_pass(ray_d, ray_o, near_plane, far_plane, app_emb,
                                                                                                     trans_emb, encode_position_function, encode_direction_function, config, use_transient)

                static_rgb_map_fine_list.append(static_rgb_map_fine)
                transient_rgb_map_fine_list.append(transient_rgb_map_fine)

            else:
                _, _, _, _, rgb_map_fine, _, _, _ = \
                    forward_pass(ray_d, ray_o, near_plane, far_plane, app_emb, trans_emb,
                                 encode_position_function, encode_direction_function, config, use_transient)

            rgb_map_fine_list.append(rgb_map_fine)

        rgb_map_fine = F.concatenate(*rgb_map_fine_list, axis=0)
        rgb_map_fine = F.reshape(rgb_map_fine, image_shape)

        if use_transient:
            static_rgb_map_fine = F.concatenate(
                *static_rgb_map_fine_list, axis=0)
            static_rgb_map_fine = F.reshape(static_rgb_map_fine, image_shape)

        frames.append((255*np.clip(rgb_map_fine.data, 0, 1)).astype(np.uint8))
        if use_transient:
            static_frames.append(
                (255*np.clip(static_rgb_map_fine.data, 0, 1)).astype(np.uint8))

    imageio.mimwrite(args.output_filename, frames, fps=30)
    imageio.mimwrite(args.output_static_filename, static_frames, fps=30)


if __name__ == '__main__':
    main()
