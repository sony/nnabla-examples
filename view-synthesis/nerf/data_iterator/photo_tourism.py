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

import glob
import sys
import numpy as np
import os
import pandas as pd
import pickle
from PIL import Image

from nnabla.utils.data_iterator import data_iterator
from nnabla.utils.data_source import DataSource


from .colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary


def get_ray_directions(H, W, K):

    i, j = np.meshgrid(np.arange(W), np.arange(H))

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    directions = \
        np.stack([(i-cx)/fx, -(j-cy)/fy, -np.ones_like(i)],
                 axis=-1)  # (H, W, 3)

    return directions


def get_rays(directions, c2w):

    rays_d = directions @ c2w[:, :3].T  # (H, W, 3)
    rays_d = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)
    rays_o = np.broadcast_to(c2w[:, 3], rays_d.shape)  # (H, W, 3)

    rays_d = rays_d.reshape(-1, 3)
    rays_o = rays_o.reshape(-1, 3)

    return rays_o, rays_d


class PhototourismDataSource(DataSource):
    def __init__(self, root_dir, split='train', img_downscale=1, val_num=1, use_cache=False, shuffle=True):

        super(PhototourismDataSource, self).__init__(shuffle=True)

        self._shuffle = shuffle
        if split != 'test':
            self._variables = ('rays', 'idx', 'rgb')
        else:
            self._variables = ('rays', 'idx')

        self.root_dir = root_dir
        self.split = split
        assert img_downscale >= 1, 'image can only be downsampled, please set img_downscale>=1!'
        self.img_downscale = img_downscale
        if split == 'val':  # image downscale=1 will cause OOM in val mode
            self.img_downscale = max(2, self.img_downscale)
        self.val_num = max(1, val_num)  # at least 1
        self.use_cache = use_cache

        self.read_meta()

        if split != 'test':
            self._size = self.__len__()

        self.reset()
        self.white_back = False

    def reset(self):
        if self._shuffle:
            self._indexes = np.arange(self._size)
            np.random.shuffle(self._indexes)
        else:
            self._indexes = np.arange(self._size)
        super(PhototourismDataSource, self).reset()

    def read_meta(self):
        # read all files in the tsv first (split to train and test later)
        tsv = glob.glob(os.path.join(self.root_dir, '*.tsv'))[0]
        self.scene_name = os.path.basename(tsv)[:-4]
        self.files = pd.read_csv(tsv, sep='\t')
        # remove data without id
        self.files = self.files[~self.files['id'].isnull()]
        self.files.reset_index(inplace=True, drop=True)

        # Step 1. load image paths
        # Attention! The 'id' column in the tsv is BROKEN, don't use it!!!!
        # Instead, read the id from images.bin using image file name!
        if self.use_cache:
            with open(os.path.join(self.root_dir, f'cache/img_ids.pkl'), 'rb') as f:
                self.img_ids = pickle.load(f)
            with open(os.path.join(self.root_dir, f'cache/image_paths.pkl'), 'rb') as f:
                self.image_paths = pickle.load(f)
        else:
            imdata = read_images_binary(os.path.join(
                self.root_dir, 'dense/sparse/images.bin'))
            img_path_to_id = {}
            for v in imdata.values():
                img_path_to_id[v.name] = v.id
            self.img_ids = []
            self.image_paths = {}  # {id: filename}
            for filename in list(self.files['filename']):
                id_ = img_path_to_id[filename]
                self.image_paths[id_] = filename
                self.img_ids += [id_]

        # Step 2: read and rescale camera intrinsics
        if self.use_cache:
            with open(os.path.join(self.root_dir, f'cache/Ks{self.img_downscale}.pkl'), 'rb') as f:
                self.Ks = pickle.load(f)
        else:
            self.Ks = {}  # {id: K}
            camdata = read_cameras_binary(os.path.join(
                self.root_dir, 'dense/sparse/cameras.bin'))
            for id_ in self.img_ids:
                K = np.zeros((3, 3), dtype=np.float32)
                cam = camdata[id_]
                img_w, img_h = int(cam.params[2]*2), int(cam.params[3]*2)
                img_w_, img_h_ = img_w//self.img_downscale, img_h//self.img_downscale

                K[0, 0] = cam.params[0]*img_w_/img_w  # fx
                K[1, 1] = cam.params[1]*img_h_/img_h  # fy
                K[0, 2] = cam.params[2]*img_w_/img_w  # cx
                K[1, 2] = cam.params[3]*img_h_/img_h  # cy
                K[2, 2] = 1
                self.Ks[id_] = K

        # Step 3: read c2w poses (of the images in tsv file only) and correct the order
        if self.use_cache:
            self.poses = np.load(os.path.join(
                self.root_dir, 'cache/poses.npy'))
        else:
            w2c_mats = []
            bottom = np.array([0, 0, 0, 1.]).reshape(1, 4)
            for id_ in self.img_ids:
                im = imdata[id_]
                R = im.qvec2rotmat()
                t = im.tvec.reshape(3, 1)
                w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
            w2c_mats = np.stack(w2c_mats, 0)  # (N_images, 4, 4)
            self.poses = np.linalg.inv(w2c_mats)[:, :3]  # (N_images, 3, 4)
            # Original poses has rotation in form "right down front", change to "right up back"
            self.poses[..., 1:3] *= -1

        # Step 4: correct scale
        if self.use_cache:
            self.xyz_world = np.load(os.path.join(
                self.root_dir, 'cache/xyz_world.npy'))
            with open(os.path.join(self.root_dir, f'cache/nears.pkl'), 'rb') as f:
                self.nears = pickle.load(f)
            with open(os.path.join(self.root_dir, f'cache/fars.pkl'), 'rb') as f:
                self.fars = pickle.load(f)
        else:
            pts3d = read_points3d_binary(os.path.join(
                self.root_dir, 'dense/sparse/points3D.bin'))
            self.xyz_world = np.array([pts3d[p_id].xyz for p_id in pts3d])
            xyz_world_h = np.concatenate(
                [self.xyz_world, np.ones((len(self.xyz_world), 1))], -1)
            # Compute near and far bounds for each image individually
            self.nears, self.fars = {}, {}  # {id_: distance}
            for i, id_ in enumerate(self.img_ids):
                # xyz in the ith cam coordinate
                xyz_cam_i = (xyz_world_h @ w2c_mats[i].T)[:, :3]
                # filter out points that lie behind the cam
                xyz_cam_i = xyz_cam_i[xyz_cam_i[:, 2] > 0]
                self.nears[id_] = np.percentile(xyz_cam_i[:, 2], 0.1)
                self.fars[id_] = np.percentile(xyz_cam_i[:, 2], 99.9)

            max_far = np.fromiter(self.fars.values(), np.float32).max()
            scale_factor = max_far/5  # so that the max far is scaled to 5
            self.poses[..., 3] /= scale_factor
            for k in self.nears:
                self.nears[k] /= scale_factor
            for k in self.fars:
                self.fars[k] /= scale_factor
            self.xyz_world /= scale_factor
        self.poses_dict = {id_: self.poses[i]
                           for i, id_ in enumerate(self.img_ids)}

        # Step 5. split the img_ids (the number of images is verfied to match that in the paper)
        self.img_ids_train = [id_ for i, id_ in enumerate(self.img_ids)
                              if self.files.loc[i, 'split'] == 'train']
        self.img_ids_test = [id_ for i, id_ in enumerate(self.img_ids)
                             if self.files.loc[i, 'split'] == 'test']
        self.N_images_train = len(self.img_ids_train)
        self.N_images_test = len(self.img_ids_test)

        if self.split == 'train':  # create buffer of all rays and rgb data
            if self.use_cache:
                self.all_rays = np.load(os.path.join(self.root_dir,
                                                     f'cache/rays{self.img_downscale}.npy'))
                # self.all_rays = torch.from_numpy(all_rays)
                self.all_rgbs = np.load(os.path.join(self.root_dir,
                                                     f'cache/rgbs{self.img_downscale}.npy'))
                # self.all_rgbs = torch.from_numpy(all_rgbs)
            else:
                self.all_rays = []
                self.all_rgbs = []
                for id_ in self.img_ids_train:
                    c2w = self.poses_dict[id_]

                    img = Image.open(os.path.join(self.root_dir, 'dense/images',
                                                  self.image_paths[id_])).convert('RGB')
                    img_w, img_h = img.size
                    if self.img_downscale > 1:
                        img_w = img_w//self.img_downscale
                        img_h = img_h//self.img_downscale
                        img = img.resize((img_w, img_h), Image.LANCZOS)
                    img = np.array(img).transpose(2, 0, 1)
                    img = img.reshape(3, -1).transpose(1, 0)
                    self.all_rgbs += [img]

                    directions = get_ray_directions(img_h, img_w, self.Ks[id_])
                    rays_o, rays_d = get_rays(directions, c2w)
                    rays_t = id_ * np.ones((len(rays_o), 1))

                    self.all_rays += [np.concatenate([rays_o, rays_d,
                                                      self.nears[id_] *
                                                      np.ones(
                                                          shape=rays_o[:, :1].shape),
                                                      self.fars[id_] *
                                                      np.ones(
                                                          shape=rays_o[:, :1].shape),
                                                      rays_t],
                                                     axis=1)]  # (h*w, 8)

                self.all_rays = np.concatenate(
                    self.all_rays, axis=0)  # ((N_images-1)*h*w, 8)
                self.all_rgbs = np.concatenate(
                    self.all_rgbs, axis=0)  # ((N_images-1)*h*w, 3)

        # use the first image as val image (also in train)
        elif self.split in ['val', 'test_train']:
            self.val_id = self.img_ids_train[0]

        else:
            pass

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        if self.split == 'test_train':
            return self.N_images_train
        if self.split == 'val':
            return self.val_num

        return len(self.poses_test)

    def _get_data(self, position):
        idx = self._indexes[position]

        if self.split == 'train':  # use data in the buffers
            # sample = {'rays': self.all_rays[idx, :8],
            #           'ts': self.all_rays[idx, 8].long(),
            #           'rgbs': self.all_rgbs[idx]}
            sample = (self.all_rays[idx, :8],
                      self.all_rays[idx, 8], self.all_rgbs[idx])

        elif self.split in ['val', 'test_train']:
            if self.split == 'val':
                id_ = self.val_id
            else:
                id_ = self.img_ids_train[idx]
            c2w = self.poses_dict[id_]

            img = Image.open(os.path.join(self.root_dir, 'dense/images',
                                          self.image_paths[id_])).convert('RGB')
            img_w, img_h = img.size
            if self.img_downscale > 1:
                img_w = img_w//self.img_downscale
                img_h = img_h//self.img_downscale
                img = img.resize((img_w, img_h), Image.LANCZOS)
            # img = self.transform(img) # (3, h, w)
            img = np.array(img).transpose(2, 0, 1).astype(np.float32)/255.0

            directions = get_ray_directions(img_h, img_w, self.Ks[id_])
            rays_o, rays_d = get_rays(directions, c2w)
            rays = np.concatenate([rays_o, rays_d,
                                   self.nears[id_] *
                                   np.ones(shape=rays_o[:, :1].shape),
                                   self.fars[id_]*np.ones(shape=rays_o[:, :1].shape)],
                                  axis=1)  # (h*w, 8)
            ts = id_*np.ones(shape=(len(rays),))

            sample = (rays, ts, img)

        else:
            sample = {}
            c2w = self.poses_test[idx]
            directions = get_ray_directions(
                self.test_img_h, self.test_img_w, self.test_K)
            rays_o, rays_d = get_rays(directions, c2w)
            near, far = 0, 5
            rays = np.concatenate([rays_o, rays_d,
                                   near*np.ones(shape=rays_o[:, :1].shape),
                                   far*np.ones(shape=rays_o[:, :1].shape)],
                                  axis=1)  # (h*w, 8)
            ts = self.test_appearance_idx*np.ones(shape=(len(rays),))
            # sample['rays'] = rays
            # sample['ts'] = F.constant(self.test_appearance_idx, shape=(len(rays),))   # torch.ones(len(rays), dtype=torch.long)
            sample = (rays, ts)

        return sample


def get_photo_tourism_dataiterator(config, split, comm):

    print(
        f'Loading {split} images downscaled by a factor of {config.data.downscale}...')
    data_source = PhototourismDataSource(config.data.root, img_downscale=int(config.data.downscale),
                                         use_cache=config.data.use_cache, split=split)
    if split == 'train':
        di = data_iterator(data_source, batch_size=config.train.ray_batch_size)
        if comm is not None:
            di_ = di.slice(
                rng=None, num_of_slices=comm.n_procs, slice_pos=comm.rank)
        else:
            di_ = di
    elif split == 'val':
        di_ = data_iterator(data_source, batch_size=1)

    elif split == 'test':
        return data_source

    return di_


if __name__ == '__main__':

    # data_source = PhototourismDataSource(sys.argv[1], img_downscale=2)
    # di = data_iterator(data_source, batch_size=256)

    # import pdb; pdb.set_trace()

    data_source = PhototourismDataSource(
        sys.argv[1], img_downscale=2, split='val')
    di = data_iterator(data_source, batch_size=1)

    data_source = PhototourismDataSource(
        sys.argv[1], img_downscale=2, use_cache=True)
    di = data_iterator(data_source, batch_size=256)

    data_source = PhototourismDataSource(
        sys.argv[1], img_downscale=2, split='val', use_cache=True)
    di = data_iterator(data_source, batch_size=1)
