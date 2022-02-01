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
import pickle
import numpy as np
from PIL import Image
import albumentations as A
from nnabla.utils.data_iterator import data_iterator_simple
from nnabla.logger import logger


def data_iterator_celeba(img_path, attributes,
                         transform=None, batch_size=32, num_samples=-1, shuffle=True, rng=None):
    """
    create celebA data iterator
    Args:
        img_path(list) : list of image paths
        attributes (dict) : attribute list
        transform : transform the image(data augmentation)
        batch_size (int) :  number of samples contained in each generated batch
        num_samples (int) : number of samples taken in data loader
                            (if num_samples=-1, it will take all the images in the dataset)
        shuffle (bool) : shuffle the data
    Returns:
        simple data iterator
    """
    imgs = img_path
    attr = attributes
    if num_samples == -1:
        num_samples = len(imgs)
    else:
        logger.info(
            "Num. of data ({}) is used for debugging".format(num_samples))

    def load_func(i):
        pillow_image = Image.open(imgs[i])
        image = np.array(pillow_image)
        transformed_image = transform(image=image)['image'].transpose(2, 0, 1)
        return transformed_image, attr[imgs[i]]

    return data_iterator_simple(load_func, num_samples,
                                batch_size, shuffle=shuffle, rng=rng, with_file_cache=False)


def actual_celeba_dataset(data_settings, batch_size, augment=True, shuffle=True, split='train'):
    """
    Create actual celebA data loader

    Args:
        data_settings (dict) : provide the data settings
        batch_size (int) : number of samples contained in each generated batch
        augment (bool) : data augmentation (True for training)
        shuffle (bool) : shuffle the data
        split (string) : split the dataset depends on the split attribute(train, valid, and test)
    Returns:
        data loader
    """

    list_ids = []
    labels = {}

    # actual celebA dataset
    attribute = data_settings['attribute']
    protected_attribute = data_settings['protected_attribute']
    train_beg = data_settings['data_params']['train_beg']
    valid_beg = data_settings['data_params']['valid_beg']
    test_beg = data_settings['data_params']['test_beg']
    img_path = data_settings['path'] + '/img_align_celeba/'
    attr_path = data_settings['path'] + '/list_attr_celeba.txt'

    label_file = open(attr_path, 'r')
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

    for i in range(beg + 2, beg + number_samples + 2):
        temp = label_file[i].strip().split()
        list_ids.append(img_path + temp[0])
        labels[img_path + temp[0]] = np.array(
            [int((int(temp[attribute + 1]) + 1) / 2),
             int((int(temp[protected_attribute + 1]) + 1) / 2)])
    # as per the author's citation, we have transformed the input image
    # (resize to 64 * 64, 256 * 256, 224 * 224)
    pre_process = [(64, 64), (256, 256), (224, 224)]
    mean_normalize = (0.485, 0.456, 0.406)
    std_normalize = (0.229, 0.224, 0.225)

    if augment:
        transform = A.Compose([
            A.Resize(pre_process[0][0], pre_process[0][1]),
            A.Resize(pre_process[1][0], pre_process[1][1]),
            A.RandomCrop(width=pre_process[2][0], height=pre_process[2][1]),
            A.HorizontalFlip(p=0.5),  # default p=0.5
            # normalize the image
            A.Normalize(mean=mean_normalize, std=std_normalize)
        ])
    else:
        transform = A.Compose([
            A.Resize(pre_process[0][0], pre_process[0][1]),
            A.Resize(pre_process[1][0], pre_process[1][1]),
            A.CenterCrop(width=pre_process[2][0], height=pre_process[2][1]),
            A.Normalize(mean=mean_normalize, std=std_normalize)
        ])
    loader = data_iterator_celeba(list_ids, labels,
                                  transform=transform, batch_size=batch_size, shuffle=shuffle)
    return loader


def fake_dataset_no_label(path, range1, batch_size=32, shuffle=False):
    """
    Create fake dataset with no label

    Args:
        path (str) : provide the data settings
        range1 (tuple) : range of generated images
        batch_size (int): number of samples contained in each generated batch
        shuffle (bool) : shuffle the data
    Returns:
        data loader
    """

    list_ids = []
    labels = {}

    for i in range(range1[0], range1[1]):
        list_ids.append(path + 'gen_'+str(i)+'.jpg')
        labels[path + 'gen_'+str(i)+'.jpg'] = -1

    # as per the author's citation, we have transformed the input image
    # (resize to 64 * 64, 256 * 256, 224 * 224)
    pre_process = [(64, 64), (256, 256), (224, 224)]
    mean_normalize = (0.485, 0.456, 0.406)
    std_normalize = (0.229, 0.224, 0.225)
    transform = A.Compose([
        A.Resize(pre_process[0][0], pre_process[0][1]),
        A.Resize(pre_process[1][0], pre_process[1][1]),
        A.CenterCrop(width=pre_process[2][0], height=pre_process[2][1]),
        A.Normalize(mean=mean_normalize, std=std_normalize)
    ])

    loader = data_iterator_celeba(list_ids, labels,
                                  transform=transform, batch_size=batch_size, shuffle=shuffle)

    return loader


def debiased_celeba_dataset(data_settings, batch_size, augment=True, split='train', shuffle=True):
    """
    Create augmentation data loader

    Args:
        data_settings (dict) : provide the data settings
        batch_size (int): number of samples contained in each generated batch
        augment (bool) : Data augmentation (True for training)
        shuffle (bool) : shuffle the data
        split (string) : split the dataset depends on the split attribute(train, valid, and test)
    Returns:
        data loader
    """

    list_ids = []
    labels = {}

    # real params configuration
    attribute = data_settings['real_params']['attribute']
    protected_attribute = data_settings['real_params']['protected_attribute']
    train_beg = data_settings['real_params']['data_params']['train_beg']
    valid_beg = data_settings['real_params']['data_params']['valid_beg']
    test_beg = data_settings['real_params']['data_params']['test_beg']
    img_path = data_settings['real_params']['path'] + '/img_align_celeba/'
    attr_path = data_settings['real_params']['path'] + '/list_attr_celeba.txt'

    label_file = open(attr_path, 'r')
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

    for i in range(beg + 2, beg + number_samples + 2):
        temp = label_file[i].strip().split()
        list_ids.append(img_path + temp[0])
        labels[img_path + temp[0]] = np.array(
            [int((int(temp[attribute + 1]) + 1) / 2),
             int((int(temp[protected_attribute + 1]) + 1) / 2)])
    # as per the author's citation, we have transformed the input image
    # (resize to 64 * 64, 256 * 256, 224 * 224)
    pre_process = [(64, 64), (256, 256), (224, 224)]
    mean_normalize = (0.485, 0.456, 0.406)
    std_normalize = (0.229, 0.224, 0.225)
    if augment:
        transform = A.Compose([
            A.Resize(pre_process[0][0], pre_process[0][1]),
            A.Resize(pre_process[1][0], pre_process[1][1]),
            A.RandomCrop(width=pre_process[2][0], height=pre_process[2][1]),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=mean_normalize, std=std_normalize)
        ])
    else:
        transform = A.Compose([
            A.Resize(pre_process[0][0], pre_process[0][1]),
            A.Resize(pre_process[1][0], pre_process[1][1]),
            A.CenterCrop(width=pre_process[2][0], height=pre_process[2][1]),
            A.Normalize(mean=mean_normalize, std=std_normalize)
        ])

    # generated image configuration
    gen_img_path = data_settings['gen_params']['generated_image_path']
    flipped_img_path = data_settings['gen_params']['flipped_images_path']
    attribute_labels = data_settings['gen_params']['label_path']
    domian_labels = data_settings['gen_params']['domain_path']
    gen_image_range = data_settings['gen_params']['flipped_image_range']
    flipped_image_range = data_settings['gen_params']['new_range']

    labeldata = pickle.load(open(attribute_labels, 'rb'))
    labeldata = labeldata[gen_image_range[0]:gen_image_range[1]]
    org_label_start = len(labeldata)
    labeldata = np.tile(labeldata, 2)

    domdata = pickle.load(open(domian_labels, 'rb'))
    domdata = domdata[gen_image_range[0]:gen_image_range[1]]
    domdata = np.concatenate([(1-domdata), domdata])

    for i in range(gen_image_range[0], gen_image_range[1]):
        list_ids.append(os.path.join(gen_img_path, 'gen_'+str(i)+'.jpg'))
        labels[os.path.join(gen_img_path, 'gen_'+str(i)+'.jpg')] = \
            np.array([labeldata[org_label_start], domdata[org_label_start]])
        org_label_start += 1

    for i in range(flipped_image_range[0], flipped_image_range[1]):
        list_ids.append(os.path.join(flipped_img_path, 'gen_'+str(i)+'.jpg'))
        labels[os.path.join(flipped_img_path, 'gen_'+str(i)+'.jpg')] =\
            np.array([labeldata[i], domdata[i]])

    loader = data_iterator_celeba(list_ids, labels,
                                  transform=transform, batch_size=batch_size, shuffle=shuffle)

    return loader
