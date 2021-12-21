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

from .llff import get_llff_data
from .synthetic import get_synthetic_data
from .deepvoxel import load_dv_data

import numpy as np


def get_data(config):

    if config['data']['name'] == 'phototourism':
        images, poses = get_phototourism_data(
            config.data.root, downscale=config.data.downscale)

    elif config['data']['name'] == 'blender':
        images, poses, render_poses, hwf, i_test = get_synthetic_data(
            config.data.root, downscale=config.data.downscale, color_perturb=config.data.color_perturb, occ_perturb=config.data.occ_perturb)
        poses = poses[:, :3, :4]

        if config['train']['white_bkgd']:
            images = images[..., :3]*images[..., -1:] + (1.-images[..., -1:])
        else:
            images = images[..., :3]

        if not isinstance(i_test, list):
            i_test = [i_test]
        if config['data']['test_hold'] > 0:
            i_test = np.arange(images.shape[0])[:: config['data']['test_hold']]
        i_train = np.array(
            [
                i
                for i in np.arange(images.shape[0])
                if (i not in i_test)
            ]
        )

        near_plane = 2
        far_plane = 6

    if config['data']['name'] == 'llff':
        images, poses, _, render_poses, i_test = get_llff_data(config['data']['root'], config['data']['factor'],
                                                               recenter=True, bd_factor=.75,
                                                               spherify=config['data']['spherify'])
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print(
            f'Loaded llff, Image Shape: {images.shape}, Pose shape: {render_poses.shape}, HWF: {hwf}')
        if not isinstance(i_test, list):
            i_test = [i_test]

        if config['data']['test_hold'] > 0:
            print('Auto LLFF holdout,', config['data']['test_hold'])
            i_test = np.arange(images.shape[0])[::config['data']['test_hold']]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])

        # Scene Discovery is done in NDC system for this data
        near_plane = 0.
        far_plane = 1.

    elif config['data']['name'] == 'deepvoxel':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=config['data']['scene_name'],
                                                                 basedir=config['data']['root'],
                                                                 testskip=config['data']['test_skip'])
        print(
            f'Loaded deepvoxels Images shape: {images.shape}, Poses shape: {render_poses.shape}, HWF: {hwf}')
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1))
        near_plane = hemi_R-1.
        far_plane = hemi_R+1.

    return images, poses, render_poses, hwf, i_test, i_train, near_plane, far_plane
