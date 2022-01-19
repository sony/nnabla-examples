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


'''
Given a Caltech101 dataset downloaded from the official page, this creates dataset files for Caltech101 training which includes;
* Image file lists for training and validation respectively, each line of which must consist of `<relative path to a image> <category index>`.
* CSV file which describe correspondence between a catgory index and a category name, each line of which consits of `<category index>,<category name>`.
'''

import os
import tqdm

# Configuration (TODO: use argparser) -------
# Location of Caltech101 dataset
caltech101_dir = '101_ObjectCategories'

# Fraction of training split. Remaining is used for validation.
frac_train = 0.8
# -------------------------------------------

categories = [(i, cat) for i, cat in enumerate(
    filter(lambda x: not x == 'Faces_easy', sorted(os.listdir(caltech101_dir))))]


def get_train_val_set_per_category(i, cat):
    cat_dir = os.path.join(caltech101_dir, cat)
    images = sorted(os.listdir(cat_dir))
    num_images = len(images)
    num_train = int(num_images * frac_train)
    num_val = num_images - num_train
    train_images = images[:num_train]
    val_images = images[num_train:]
    assert num_val == len(val_images)
    return train_images, val_images


with open('caltech101_train.txt', 'w') as tfd, open('caltech101_val.txt', 'w') as vfd, \
        open('caltech101_categories.csv', 'w') as cfd:
    for i, cat in tqdm.tqdm(categories):
        train_images, val_images = get_train_val_set_per_category(i, cat)
        for img in train_images:
            print(f'{cat}/{img} {i}', file=tfd)
        for img in val_images:
            print(f'{cat}/{img} {i}', file=vfd)
        print(f'{i},{cat}', file=cfd)
print('done')
