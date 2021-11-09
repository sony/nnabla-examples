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
import numpy as np
from nnabla.utils.data_iterator import data_iterator
from nnabla.utils.data_source import DataSource
from nnabla.utils import image_utils

import os
import glob
import imageio

from helper import load_K_Rt_from_P, generate_raydir_camloc


class DTUMVSDataSource(DataSource):
    '''
    Load DTUMVS dataset from the zip file created by the author of "Multiview Neural Surface Reconstruction by Disentangling Geometry and Appearance".
    '''

    def _get_data(self, position):
        img_idx = self._img_indices[position]
        image = self._images[img_idx]
        mask = self._masks[img_idx]
        intrinsic = self._intrinsics[img_idx]
        pose = self._poses[img_idx]

        H, W, _ = image.shape
        R = H * W

        color = image.reshape((R, 3))[self.pixel_idx]
        mask = mask.reshape((R, 1))[self.pixel_idx]
        xy = self._xy[self.pixel_idx]
        return color, mask, intrinsic, pose, xy

    def _load_dtumvs(self, path):
        # Images
        image_files = sorted(glob.glob(os.path.join(path, "image", "*")))
        images = np.asarray([image_utils.imread(f) for f in image_files])
        images = images * (1.0 / 127.5) - 1.0

        # Masks
        mask_files = sorted(glob.glob(os.path.join(path, "mask", "*")))
        masks = np.asarray([imageio.imread(f, as_gray=True)[:, :, np.newaxis] > 127.5
                            for f in mask_files]) * 1.0

        # Camera projection matrix and scale matrix for special correctness
        cameras = np.load(os.path.join(path, "cameras.npz"))
        world_mats = [cameras['world_mat_%d' % idx].astype(
            np.float32) for idx in range(len(images))]
        scale_mats = [cameras['scale_mat_%d' % idx].astype(
            np.float32) for idx in range(len(images))]

        intrinsics, poses = [], []
        for W, S in zip(world_mats, scale_mats):
            P = W @ S
            P = P[:3, :4]
            intrinsic, pose = load_K_Rt_from_P(P)
            intrinsics.append(intrinsic[:3, :3])
            poses.append(pose)

        # return images[0:1, ...], masks[0:1, ...], np.asarray(intrinsics)[0:1, ...], np.asarray(poses)[0:1, ...]
        return images, masks, np.asarray(intrinsics), np.asarray(poses)

    def __init__(self, path, n_rays, train=True, shuffle=False, rng=None):
        super(DTUMVSDataSource, self).__init__(shuffle=shuffle)
        self._n_rays = n_rays
        self._train = train

        self._images, self._masks, self._intrinsics, self._poses = self._load_dtumvs(
            path)

        # assume all images have same resolution
        H, W, _ = self._images[0].shape
        x = np.arange(W)
        y = np.arange(H)
        xx, yy = np.meshgrid(x, y)
        self._xy = np.asarray([xx.flatten(), yy.flatten()]).T

        self._size = len(self._images)
        self._pixels = H * W
        self._variables = ('image', 'mask', 'intrinsic', 'pose', 'xy')
        if rng is None:
            rng = np.random.RandomState(313)
        self.rng = rng
        self.reset()

        self.pixel_idx = self._sampling_idx()

        dname = os.path.split(path.rstrip("/"))[-1]
        nn.logger.info(f"--- Finish loading DTU MVS dataset ({dname}). ---")
        nn.logger.info(f"Num. of images = {self._size}")
        nn.logger.info(f"Num. of pixels (H x W) = {self._pixels} ({H} x {W})")
        nn.logger.info(f"Num. of random rays = {self._n_rays}")

    def reset(self):
        if self._shuffle:
            self._img_indices = self.rng.permutation(self._size)
        else:
            self._img_indices = np.arange(self._size)
        super(DTUMVSDataSource, self).reset()

    @property
    def images(self):
        """Get copy of whole data with a shape of (B, H, W, 3)."""
        return self._images.copy()

    @property
    def poses(self):
        return self._poses.copy()

    @property
    def intrinsics(self):
        return self._intrinsics.copy()

    @property
    def masks(self):
        return self._masks.copy()

    def change_sampling_idx(self):
        self.pixel_idx = self._sampling_idx()

    def _sampling_idx(self):
        return self.rng.randint(0, self._pixels, self._n_rays)


def data_iterator_dtumvs(data_source,
                         batch_size,
                         rng=None,
                         with_memory_cache=False,
                         with_file_cache=False):
    '''
    Provide DataIterator with :py:class:`DTUMVSDataSource`
    with_memory_cache and with_file_cache option's default value is all False,
    because :py:class:`DTUMVSDataSource` is able to store all data into memory.
    '''
    return data_iterator(data_source,
                         batch_size,
                         rng,
                         with_memory_cache,
                         with_file_cache)


def main(args):
    # Data Iterator
    ds = DTUMVSDataSource(args.path, args.n_rays, shuffle=True)
    di = data_iterator_dtumvs(ds, 1)
    for i in range(args.iters):
        pcolor, mask, intrinsic, pose, xy = di.next()
        print(f"pcolor.shape = {pcolor.shape}")
        print(f"mask.shape = {mask.shape}")
        print(f"intrinsic.shape = {intrinsic.shape}")
        print(f"pose.shape = {pose.shape}")
        print(f"xy.shape = {xy.shape}")
        print(f"Pcolor (min, max) = ({pcolor.min()}, {pcolor.max()})")
        print(f"Mask (min, max) = ({mask.min()}, {mask.max()})")

    # Generate rays
    raydir, camloc = generate_raydir_camloc(pose, intrinsic, xy)
    print(f"raydir.shape = {raydir.shape}")
    np.testing.assert_allclose(
        np.sum(raydir ** 2, axis=-1) ** 0.5, 1.0, atol=1e-6)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="DTU MVS Dataset.")
    parser.add_argument('--path', type=str, required=True,
                        help="Path to scale directory, Ex, DTU/scan24.")
    parser.add_argument('--batch-size', '-b', type=int, default=4)
    parser.add_argument('--n-rays', '-n', type=int, default=800)
    parser.add_argument('--iters', '-i', type=int, default=100)

    args = parser.parse_args()
    main(args)
