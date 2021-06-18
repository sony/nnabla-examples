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
import os
import numpy as np
from nnabla.utils.image_utils import imread, imresize


def scan_file(filename, delimiter):
    import re
    f = open(filename)
    strmat = []
    for line in f:
        line = line[:-1]
        strmat.append(re.split(delimiter, line))
    f.close()
    return strmat


def get_filenames(ca_filenames, ca_ids, cid, max_files=100000):
    cids = np.where(ca_ids == cid)[0]
    filenames = []
    for i in range(min(len(cids), max_files)):
        filename = "img_align_celeba/" + ca_filenames[cids[i]]
        filenames.append(filename)
    return filenames


def get_sliced_images(filenames, resize=True):
    xs = []
    for filename in filenames:
        x = imread(filename)
        x = x[45:173, 25:153, :]
        if resize:
            x = imresize(x, size=(64, 64), interpolate='lanczos')
        xs.append(x)
    return xs


if not os.path.exists('celeb_a/data'):
    os.makedirs('celeb_a/data')


identity_celeb_a = scan_file("identity_CelebA.txt", " ")
ca_array = np.asarray(identity_celeb_a)
ca_filenames = ca_array[:, 0]
ca_ids = np.asarray(list(map(int, ca_array[:, 1])))

xs = []
ys = []
n = max(ca_ids) - 1
for cid in range(n):
    filenames = get_filenames(ca_filenames, ca_ids, cid + 1)
    x = get_sliced_images(filenames, resize=True)
    y = np.ones(len(filenames), dtype=np.int) * cid
    xs.extend(x)
    ys.extend(y)

xs = np.array(xs)
ys = np.array(ys)

np.save("celeb_a/data/celeb_images.npy", xs)
np.save("celeb_a/data/celeb_labels.npy", ys)
