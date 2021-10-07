# Copyright (c) 2021 Sony Group Corporation. All Rights Reserved.
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

from __future__ import absolute_import

import matplotlib.pyplot as plt
import numpy as np

score = np.load("./data/info/shuffle/alpha_vgg_nnabla_score.npy")
y_raw = np.load("./data/input/shuffle/y_train.npy")
y_shuffle = np.load("./data/input/shuffle/y_shuffle_train.npy")

sort_ind = np.argsort(score)[::-1]
y_raw_sort = y_raw[sort_ind]
y_shuffle_sort = y_shuffle[sort_ind]

mislabel_length = len(np.where(y_raw != y_shuffle)[0])
print("mislabel: ", mislabel_length)
extracted = np.array([])
count = 0
for raw, shuffle in zip(y_raw_sort, y_shuffle_sort):
    if raw != shuffle:
        count += 1
    extracted = np.append(extracted, count)

detection_rate = extracted / mislabel_length
fraction_rate = [i / len(y_raw) for i in range(len(y_raw))]


fontsize = 16
plt.figure()
plt.plot(fraction_rate, detection_rate)
plt.xlabel("fraction of data checked", fontsize=fontsize)
plt.grid()
plt.ylabel("fraction of detected mislabel", fontsize=fontsize)
plt.savefig("./data/info/shuffle/mislabel_vgg_nnabla_detect.png")
