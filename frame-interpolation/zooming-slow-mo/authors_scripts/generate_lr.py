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
import argparse
import glob
import cv2
import numpy as np
from data_utils import imresize_np

#- Command line arguments -#
data_parser = argparse.ArgumentParser(description='Training data preperation')
data_parser.add_argument('--dataset-path', type=str, default="./vimeo_septuplet/sequences",
                         help='add path where vimeo dataset resides')
data_parser.add_argument('--scale', type=int, default=4,
                         help='by how much factor you want to scale the images')

args = data_parser.parse_args()


def generate_lr(source_dir, scale):
    """
    params: upscale factor, input directory, output directory
    """
    dest_lr_path = source_dir + '_LR'

    print(source_dir)
    if not os.path.isdir(source_dir):
        raise Exception('Dataset does not exist')

    if not os.path.isdir(dest_lr_path):
        os.mkdir(dest_lr_path)

    # Gather all absoulte paths of images in vimeo dataset
    filepaths = [f for f in glob.glob(
        source_dir + '/**/*.png', recursive=True)]

    num_files = len(filepaths)
    print("number of files:", num_files)

    # Creating folder and subfolders under destination image folder
    for folder in [f for f in os.listdir(source_dir)]:
        os.mkdir(os.path.join(dest_lr_path, folder))
        for sub_folder in [f for f in os.listdir(os.path.join(source_dir, folder))]:
            os.mkdir(os.path.join(dest_lr_path, folder, sub_folder))

    # ------ Prepare data with augementation -------
    for i in range(num_files):
        filename = filepaths[i]
        print('No.{} -- Processing {}'.format(i, filename))
        image = cv2.imread(filename)
        folder, sub_folder, img_name = filename.split('/')[-3:]
        image_lr = imresize_np(image, 1 / scale, True)

        # Saving generated LR frames under destination image folder
        cv2.imwrite(os.path.join(dest_lr_path, folder,
                                 sub_folder, img_name), image_lr)


if __name__ == "__main__":
    source_dir = args.dataset_path
    scale = args.scale
    generate_lr(source_dir, scale)
