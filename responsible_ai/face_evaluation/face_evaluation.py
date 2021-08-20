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

import numpy as np
import cv2


def extract_non_black(img):
    img_total = img[:, :, 0] + img[:, :, 1] + img[:, :, 2]
    non_black = np.where(img_total > 0)
    extract_img = []
    for i in range(len(non_black[0])):
        extract_img.append(img[non_black[0][i], non_black[1][i], :])
    extract_img = np.array(extract_img).reshape(1, -1, 3)
    return extract_img


def calc_ita(masked_imgs):
    itas = []
    for masked_img in masked_imgs:
        lab_img = cv2.cvtColor(masked_img, cv2.COLOR_RGB2Lab)
        l_img, a_img, b_img = cv2.split(lab_img)
        l_img = l_img * (100/255)
        b_img = b_img - 128
        ITA_img = (np.arctan((l_img-50)/b_img)*180)/np.pi
        a_hist, a_bins = np.histogram(ITA_img, bins=1000)
        ITA_each = a_bins[a_hist.argmax()]
        itas.append(ITA_each)
    return round(sum(itas)/len(itas), 2)
