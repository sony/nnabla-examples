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

import argparse
import os
import random


def get_args():
    parser = argparse.ArgumentParser(
        description='TSV file generation for Phototourism dataset')

    parser.add_argument('--data-path', '-d', type=str,
                        required=True,
                        help='Path to train and test image folders')
    parser.add_argument('--data-name', '-n', type=str,
                        required=True,
                        help='Dataset Name')
    parser.add_argument('--train-split', '-s', type=float,
                        default=0.9,
                        help='Dataset Name')
    parser.add_argument('--out-path', '-o', type=str,
                        required=True,
                        help='Output File path')

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()

    filelist = os.listdir(args.data_path)
    random.shuffle(filelist)
    train_list = filelist[:int(len(filelist)*args.train_split)]
    test_list = filelist[len(train_list):]

    print(
        f'Total: {len(filelist)} Train: {len(train_list)} Test: {len(test_list)}')

    with open(f'{args.out_path}', 'w') as f:
        f.write('filename	id	split	dataset\n')

        for i, filename in enumerate(train_list):
            f.write(f'{filename}	{i}	train	{args.data_name}\n')

        for i, filename in enumerate(test_list):
            f.write(f'{filename}	{i+len(train_list)}	test	{args.data_name}\n')

    print(f'Saved tsv file at {args.out_path}')
