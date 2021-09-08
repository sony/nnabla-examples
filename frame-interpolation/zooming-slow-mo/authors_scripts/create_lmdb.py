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

import os
import sys
import os.path as osp
import glob
import argparse
import pickle
import lmdb
import cv2
from tqdm import trange

#- Command line arguments -#
data_parser = argparse.ArgumentParser(description='dataset path')

data_parser.add_argument('--dataset-path', type=str, default='./vimeo_septuplet/sequences',
                         help='add path where files(HR or LR frames) reside')

data_parser.add_argument('--save-lmdb-path', type=str, default='./lmdb_data',
                         help='where the lmdb dataset will be saved')

data_parser.add_argument('--train-list-file', type=str, default='./vimeo_septuplet/sep_trainlist.txt',
                         help='file containing list of training images')

data_parser.add_argument('--mode', type=str, default='GT',
                         help='Create LR or GT dataset based on mode value')

args = data_parser.parse_args()


def lmdb_create(img_folder, lmdb_save_path, train_list_file, mode):
    """
    Create lmdb for the Vimeo90K-7 frames dataset, each image with fixed size
    GT: [3, 256, 448]
        Only need the 4th frame currently, e.g., 00001_0001_4
    LR: [3, 64, 112]
        With 1st - 7th frames, e.g., 00001_0001_1, ..., 00001_0001_7
    key:
        Use the folder and subfolder names, w/o the frame index, e.g., 00001_0001
    """
    # Configurations
    batch = 3000  # Change depending on your mem size
    if mode == 'GT':
        h_dst, w_dst = 256, 448
    elif mode == 'LR':
        h_dst, w_dst = 64, 112

    # Whether the lmdb file exist
    if not os.path.isdir(lmdb_save_path):
        os.mkdir(lmdb_save_path)

    # Read all the image paths to a list
    print('Reading image path list ...')
    with open(train_list_file) as f:
        file_paths = f.readlines()
        file_paths = [v.strip() for v in file_paths]
    all_img_list = []
    keys = []

    for line in file_paths:
        folder = line.split('/')[0]
        sub_folder = line.split('/')[1]
        file_l = glob.glob(osp.join(img_folder, folder, sub_folder) + '/*')
        all_img_list.extend(file_l)
        for j in range(7):
            keys.append('{}_{}_{}'.format(folder, sub_folder, j + 1))
    all_img_list = sorted(all_img_list)
    keys = sorted(keys)

    print('Calculating the total size of images...')
    data_size = sum(os.stat(v).st_size for v in all_img_list)

    # Create lmdb environment
    env = lmdb.open(osp.join(lmdb_save_path, 'Vimeo7_train_{}.lmdb'.format(
        mode)), map_size=data_size * 10)
    # txn is a Transaction object
    txn = env.begin(write=True)

    for i in trange(len(all_img_list)):
        key = keys[i]
        path = all_img_list[i]
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        key_byte = key.encode('ascii')
        # Fixed shape
        height, width, channels = img.shape
        assert height == h_dst and width == w_dst and channels == 3, 'different shape.'
        txn.put(key_byte, img)
        if i % batch == 1:
            txn.commit()
            txn = env.begin(write=True)

    txn.commit()
    env.close()

    print('Finish reading and writing {} images.'.format(len(all_img_list)))
    print('Finish writing lmdb.')


if __name__ == "__main__":
    # Function to create lmdb daatset from vimeo dataset
    lmdb_create(args.dataset_path, args.save_lmdb_path,
                args.train_list_file, args.mode)
