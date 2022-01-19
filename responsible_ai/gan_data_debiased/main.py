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
import sys
from nnabla.ext_utils import get_extension_context
import nnabla as nn
import args
import data_loader as dl
import classifier as clf
from utils import utils


def model_train_setting(opt):
    """
    Get the model train settings
    Args:
        opt : variables that containing values for all of your options
    Returns:
        variables which you need to train
    """
    attr_list = utils.get_all_attr()
    if opt['model_train'] == 'baseline':
        data_params = {
            "train_beg": opt['train_beg'],
            "valid_beg": opt['valid_beg'],
            "test_beg": opt['test_beg'],
        }
        data_setting = {
            'path': opt['base_img_path'],
            'protected_attribute': opt['protected_attribute'],
            'attribute': opt['attribute'],
            'data_params': data_params,
            'batch_size': opt['batch_size'],
            'learning_rate': opt['learning_rate'],
            'max_iter': opt['max_iter_base']
        }
        opt['data_setting'] = data_setting

    if opt['model_train'] == 'gan_debiased':
        data_params = {
            "train_beg": opt['train_beg'],
            "valid_beg": opt['valid_beg'],
            "test_beg": opt['test_beg'],
        }
        real_params = {
            'path': opt['base_img_path'],
            'attribute': opt['attribute'],
            'protected_attribute': opt['protected_attribute'],
            'data_params': data_params
        }
        generated_images = "{}/AllGenImages".format(opt["fake_data_dir"])
        flipped_images = "{}/{}/".format(opt["fake_data_dir"],
                                         attr_list[opt['attribute']])
        label_score = "{}/all_{}_scores.pkl".format(opt['fake_data_dir'],
                                                    attr_list[opt['attribute']])
        domain_score = "{}/all_{}_scores.pkl".format(opt['fake_data_dir'],
                                                     attr_list[opt['protected_attribute']])

        generated_params = {
            'generated_image_path': generated_images,
            'flipped_images_path': flipped_images,
            'label_path': label_score,
            'domain_path': domain_score,
            # flipped the images from 15000 to 175000
            'flipped_image_range': (15000, 175000),
            'orig_label_range': (160000, 320000),  # original label range
            'new_range': (0, 160000),  # new images
        }
        data_setting = {
            'real_params': real_params,
            'gen_params': generated_params,
            'batch_size': opt['batch_size'],
            'learning_rate': opt['learning_rate'],
            'max_iter': opt['max_iter_gan_debiased']
        }
        opt['data_setting'] = data_setting

    return opt


def main():
    """
    main method
    """

    opt = args.get_args()
    opt = model_train_setting(opt)
    ctx = get_extension_context(
        opt['context'], device_id=opt['device_id'], type_config=opt['type_config'])

    nn.set_default_context(ctx)
    # model configurations
    batch_size = opt['data_setting']['batch_size']
    learning_rate = opt['data_setting']['learning_rate']
    max_iter = opt['data_setting']['max_iter']
    if (opt["model_train"] == 'baseline'):
        train = dl.actual_celeba_dataset(opt['data_setting'], batch_size,
                                         augment=True, split='train', shuffle=True)
        val = dl.actual_celeba_dataset(opt['data_setting'], batch_size,
                                       augment=False, split='valid', shuffle=False)
        val_weight = None

    elif (opt["model_train"] == 'gan_debiased'):
        train = dl.debiased_celeba_dataset(opt['data_setting'], batch_size,
                                           augment=True, split='train', shuffle=True)
        val = dl.actual_celeba_dataset(opt['data_setting']['real_params'], batch_size,
                                       augment=False, split='valid', shuffle=False)
        val_weight = utils.compute_class_weight(val)
    else:
        print("please provide proper argument")
        sys.exit(0)

    attr_list = utils.get_all_attr()
    if not os.path.exists(opt['model_save_path']):
        os.makedirs(opt['model_save_path'])
    monitor_path = os.path.join(
        opt['model_save_path'], attr_list[opt['attribute']])
    if not os.path.exists(monitor_path):
        os.makedirs(monitor_path)

    attribute_classifier_model = clf.attribute_classifier(batch_size=batch_size,
                                                          learning_rate=learning_rate,
                                                          max_iter=max_iter,
                                                          monitor_path=monitor_path,
                                                          val_weight=val_weight)
    attribute_classifier_model.train(train, val)


if __name__ == '__main__':
    main()
