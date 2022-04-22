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
import numpy as np
from PIL import Image
import albumentations as A
from nnabla.utils.data_iterator import data_iterator_simple


def data_iterator_celeba(img_path, attr_path, batch_size,
                         target_attribute='Attractive', protected_attribute='Male',
                         num_samples=-1, augment=False, shuffle=False, rng=None):
    """
    create celebA data iterator
    Args:
        img_path (str) : image path directory
        attr_path (str) : celebA attribute file path (ex: list_attr_celeba.txt)
        batch_size (int) :  number of samples contained in each generated batch
        target_attribute (str) : target attribute (ex: Arched EyeBrows,
                                Bushy Eyebrows, smiling,etc..)
        protected_attribute (str): protected attribute (ex: Male, Pale_Skin)
        num_samples (int) : number of samples taken in data loader
                            (if num_samples=-1, it will take all the images in the dataset)
        augment (bool) : data augmentation (True for training)
        shuffle (bool) : shuffle the data (True /False)
        rng : None
    Returns:
        simple data iterator
    """

    imgs = []
    for file in sorted(os.listdir(img_path), key=lambda x: int(x.split(".")[0])):
        imgs.append(os.path.join(img_path, file))
    with open(attr_path, 'r') as f:
        lines = f.readlines()

    attr_list = lines[1].strip().split()
    attr_idx_dict = {attr: i for i, attr in enumerate(attr_list)}
    labels_dict = {}
    for line in lines[2:]:
        line = line.strip().split()
        key = line[0]
        attr = line[1:]
        labels_dict[key] = np.array([int((int(attr[attr_idx_dict[target_attribute]]) + 1) / 2),
                                     int((int(attr[attr_idx_dict[protected_attribute]]) + 1) / 2)])

    # as per the author's citation, we have transformed the input image
    # (resize to , 256 * 256, 224 * 224)
    pre_process = [(256, 256), (224, 224)]
    mean_normalize = (0.485, 0.456, 0.406)
    std_normalize = (0.229, 0.224, 0.225)

    if augment:
        transform = A.Compose([
            A.Resize(pre_process[0][0], pre_process[0][1]),
            A.RandomCrop(width=pre_process[1][0], height=pre_process[1][1]),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=mean_normalize, std=std_normalize)
        ])
    else:
        transform = A.Compose([
            A.Resize(pre_process[0][0], pre_process[0][1]),
            A.CenterCrop(width=pre_process[1][0], height=pre_process[1][1]),
            A.Normalize(mean=mean_normalize, std=std_normalize)
        ])
    if num_samples == -1:
        num_samples = len(imgs)
    else:
        print("Num. of data ({}) is used for debugging".format(num_samples))

    def load_func(i):
        img = Image.open(imgs[i])
        img = np.array(img.convert('RGB'))
        # transform
        transformed_image = transform(image=img)['image'].transpose(2, 0, 1)
        return transformed_image, labels_dict[os.path.basename(imgs[i])]

    return data_iterator_simple(load_func, num_samples, batch_size,
                                shuffle=shuffle, rng=rng, with_file_cache=False)
