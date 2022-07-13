# Copyright 2022 Sony Group Corporation.
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
import argparse
import shutil


def parse_args():
    """
    parse args
    """
    description = "Split the CelebA dataset "
    parser = argparse.ArgumentParser(description)
    parser.add_argument("--img_path", type=str, default=r'./data/celeba/images',
                        help="Source image path directory")
    parser.add_argument("--attr_path", type=str, default=r'./data/celeba/list_attr_celeba.txt',
                        help="celebA attribute file path (ex: list_attr_celeba.txt)")
    parser.add_argument("--out_dir", type=str, default=r'./train',
                        help='Path where the split data to be saved')
    parser.add_argument("--split", type=str, default='train',
                        help='split the dataset depends on the '
                             'split attribute(train, valid, and test)')
    opt = vars(parser.parse_args())

    return opt


def split_celeba_dataset(img_path, attr_path, out_dir, split="test"):
    """
    split the celebA dataset
    Args:
        img_path (str) : image path directory
        attr_path (str) : celebA attribute file path (ex: list_attr_celeba.txt)
        out_dir (str) : Path where the split data to be saved
        split (string) : split the dataset depends on the split attribute(train, valid, and test)
    """
    # split dataset like authors
    train_beg = 0  # train starts from
    valid_beg = 162770  # valid starts from
    test_beg = 182610  # test starts from

    with open(attr_path, 'r') as label_file:
        label_file = label_file.readlines()

    # skipping the first two rows for header
    total_samples = len(label_file) - 2
    if split == 'train':
        number_samples = valid_beg - train_beg
        beg = train_beg

    elif split == 'valid':
        number_samples = test_beg - valid_beg
        beg = valid_beg

    elif split == 'test':
        number_samples = total_samples - test_beg
        beg = test_beg
    else:
        print('Error')
        return
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for i in range(beg + 2, beg + number_samples + 2):
        temp = label_file[i].strip().split()
        src_dir = os.path.join(img_path, temp[0])
        dst_dir = os.path.join(out_dir, temp[0])
        shutil.copy(src_dir, dst_dir)
    print("splitting completed")


if __name__ == '__main__':
    opt = parse_args()
    split_celeba_dataset(opt['img_path'], opt['attr_path'],
                         opt['out_dir'], opt['split'])
