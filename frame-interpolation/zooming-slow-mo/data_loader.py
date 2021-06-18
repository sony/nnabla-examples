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

import random
import pickle
import lmdb
import numpy as np
from nnabla.utils.data_iterator import data_iterator_simple


def data_iterator(conf, shuffle, rng=None):
    """
    Data iterator for Zooming SloMo training
    return:
    """
    assert conf.data.n_frames > 1, 'Error: Not enough LR frames to interpolate'
    half_n_frames = conf.data.n_frames // 2

    # determine the LQ frame list
    # N | frames
    # 1 | error
    # 3 | 0,2
    # 5 | 0,2,4
    # 7 | 0,2,4,6

    lr_index_list = [i * 2 for i in range(1 + half_n_frames)]

    paths_gt = pickle.load(open(conf.data.cache_keys, 'rb'))

    gt_lmdb = lmdb.open(conf.data.lmdb_data_gt, readonly=True, lock=False, readahead=False,
                        meminit=False)
    lq_lmdb = lmdb.open(conf.data.lmdb_data_lq, readonly=True, lock=False, readahead=False,
                        meminit=False)

    center_frame_idx = random.randint(2, 6)  # 2<= index <=6

    def determine_neighbor_list(central_frame_idx):
        """
        given central frame index, determine neighborhood frames
        """
        interval = random.choice(conf.data.interval_list)

        if conf.data.border_mode:
            direction = 1  # 1: forward; 0: backward
            if conf.random_reverse and random.random() < 0.5:
                direction = random.choice([0, 1])
            if central_frame_idx + interval * (conf.data.n_frames - 1) > 7:
                direction = 0
            elif central_frame_idx - interval * (conf.data.n_frames - 1) < 1:
                direction = 1
            # get the neighbor list
            if direction == 1:
                neighbor_list = list(
                    range(central_frame_idx, central_frame_idx + interval * conf.data.n_frames,
                          interval))
            else:
                neighbor_list = list(
                    range(central_frame_idx, central_frame_idx - interval * conf.data.n_frames,
                          -interval))
        else:
            # ensure not exceeding the borders
            while (central_frame_idx + half_n_frames * interval > 7) or \
                    (central_frame_idx - half_n_frames * interval < 1):
                central_frame_idx = random.randint(2, 6)

            # get the neighbor list
            neighbor_list = list(
                range(central_frame_idx - half_n_frames * interval,
                      central_frame_idx + half_n_frames * interval + 1, interval))
            if conf.data.random_reverse and random.random() < 0.5:
                neighbor_list.reverse()

        return neighbor_list

    neighbors = determine_neighbor_list(center_frame_idx)
    lq_frames_list = [neighbors[i] for i in lr_index_list]

    assert len(neighbors) == conf.data.n_frames, \
        'Wrong length of neighbor list: {}'.format(len(neighbors))

    # image read and augment functions

    def augment(img_list, flip=True, rot=True):
        # flip OR rotate
        def _augment(img):
            if flip and random.random() < 0.5:
                # horizontal flip
                img = img[:, ::-1, :]
            if rot and random.random() < 0.5:
                # vertical flip and 90 degree rotation
                img = img[::-1, :, :]
                img = img.transpose(1, 0, 2)
            return img

        return [_augment(img) for img in img_list]

    def _read_img_from_lmdb(env, key, size):
        """
        read image from lmdb with key (w/ and w/o fixed size)
        size: (channels, height, width) tuple
        """
        with env.begin(write=False) as txn:
            buf = txn.get(key.encode('ascii'))
        img_flat = np.frombuffer(buf, dtype=np.uint8)
        channels, height, width = size
        img = img_flat.reshape(height, width, channels)
        img = img.astype(np.float32) / 255.
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        # some images have 4 channels
        if img.shape[2] > 3:
            img = img[:, :, :3]
        return img

    def load_zoomingslomo_data(i):
        """
        loads data, given the index -> primary function in data loader
        """
        key = paths_gt[i]

        # get the GT image (as the center frame)
        img_gt_l = [_read_img_from_lmdb(gt_lmdb, key + '_{}'.format(v), (3, 256, 448))
                    for v in neighbors]

        # get Low Quality images
        lq_size_tuple = (3, 64, 112)
        img_lq_l = [_read_img_from_lmdb(lq_lmdb, key + '_{}'.format(v), lq_size_tuple)
                    for v in lq_frames_list]

        _, height, width = lq_size_tuple  # LQ size
        # randomly crop
        scale = 4
        gt_size = conf.data.gt_size
        lr_size = gt_size // scale
        rnd_h = random.randint(0, max(0, height - lr_size))
        rnd_w = random.randint(0, max(0, width - lr_size))
        img_lq_l = [v[rnd_h:rnd_h + lr_size, rnd_w:rnd_w + lr_size, :]
                    for v in img_lq_l]
        rnd_h_highres, rnd_w_highres = int(rnd_h * scale), int(rnd_w * scale)
        img_gt_l = [
            v[rnd_h_highres:rnd_h_highres + gt_size,
                rnd_w_highres:rnd_w_highres + gt_size, :]
            for v in img_gt_l]

        # augmentation - flip, rotate
        img_lq_l = img_lq_l + img_gt_l
        rlt = augment(img_lq_l, conf.data.use_flip, conf.data.use_rot)
        img_lq_l = rlt[0:-conf.data.n_frames]
        img_gt_l = rlt[-conf.data.n_frames:]

        # stack LQ and GT images in NHWC order, N is the frame number
        img_lq_stack = np.stack(img_lq_l, axis=0)
        img_gt_stack = np.stack(img_gt_l, axis=0)

        # numpy to tensor
        img_gt_stack = img_gt_stack[:, :, :, [2, 1, 0]]  # BGR to RGB
        img_lq_stack = img_lq_stack[:, :, :, [2, 1, 0]]  # BGR to RGB
        img_gt_stack = np.ascontiguousarray(
            np.transpose(img_gt_stack, (0, 3, 1, 2)))  # HWC to CHW
        img_lq_stack = np.ascontiguousarray(
            np.transpose(img_lq_stack, (0, 3, 1, 2)))  # HWC to CHW

        return img_lq_stack, img_gt_stack

    def load_slomo_data(i):
        """
        loads data, given the index -> primary function in data loader
        """
        key = paths_gt[i]

        gt_size_tuple = (3, 256, 448)
        # get the GT image (as the center frame)
        img_gt_l = [_read_img_from_lmdb(gt_lmdb, key + '_{}'.format(v), gt_size_tuple)
                    for v in neighbors]

        _, height, width = gt_size_tuple  # GT size
        # randomly crop
        gt_size = conf.data.gt_size
        rnd_h = random.randint(0, max(0, height - gt_size))
        rnd_w = random.randint(0, max(0, width - gt_size))

        img_gt_l = [
            v[rnd_h:rnd_h + gt_size,
                rnd_w:rnd_w + gt_size, :]
            for v in img_gt_l]

        # augmentation - flip, rotate
        img_gt_l = augment(img_gt_l, conf.data.use_flip, conf.data.use_rot)

        # stack LQ and GT images in NHWC order, N is the frame number
        img_gt_stack = np.stack(img_gt_l, axis=0)
        # numpy to tensor
        img_gt_stack = img_gt_stack[:, :, :, [2, 1, 0]]  # BGR to RGB
        img_gt_stack = np.ascontiguousarray(
            np.transpose(img_gt_stack, (0, 3, 1, 2)))  # HWC to CHW

        return _, img_gt_stack

    dataset_load_func = load_zoomingslomo_data if not conf.train.only_slomo else load_slomo_data

    return data_iterator_simple(dataset_load_func, len(paths_gt), conf.train.batch_size,
                                shuffle=shuffle, rng=rng, with_file_cache=False,
                                with_memory_cache=False)
