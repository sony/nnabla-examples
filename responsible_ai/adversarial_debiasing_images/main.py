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

import args
import numpy as np
import nnabla as nn
from nnabla.ext_utils import get_extension_context
import adversarial as ad
import basemodel as bm
import data_loader as di
from utils import utils


def main():
    """
    main method
    """

    opt = args.get_args()
    ctx = get_extension_context(
        opt['context'], device_id=opt['device_id'], type_config=opt['type_config'])

    nn.set_default_context(ctx)
    # model configurations
    batch_size = opt['batch_size']
    learning_rate = opt['learning_rate']
    max_iter = opt['max_iter']
    total_epochs = opt['total_epochs']
    monitor_path = opt['model_save_path']
    rng = np.random.RandomState(1)
    train_loader = di.data_iterator_celeba(
        opt['celeba_image_train_dir'], opt['attr_path'], batch_size,
        target_attribute=opt['attribute'], protected_attribute=opt['protected_attribute'],
        augment=True, shuffle=True, rng=rng)
    valid_loader = di.data_iterator_celeba(
        opt['celeba_image_valid_dir'], opt['attr_path'], batch_size,
        target_attribute=opt['attribute'], protected_attribute=opt['protected_attribute'],
        rng=rng)
    validation_weight = utils.compute_class_weight(valid_loader)
    if opt["model_train"] == 'adversarial':
        training_ratio = opt['training_ratio']
        lamda = opt['lamda']
        adv = ad.adversarial(batch_size=batch_size, total_epochs=total_epochs,
                             learning_rate=learning_rate, max_iter=max_iter,
                             training_ratio=training_ratio, lamda=lamda,
                             monitor_path=monitor_path, validation_weight=validation_weight)
        adv.train(train_loader, valid_loader)
    elif opt["model_train"] == 'baseline':
        clf = bm.attribute_classifier(batch_size=batch_size, total_epochs=total_epochs,
                                      learning_rate=learning_rate,
                                      max_iter=max_iter,
                                      validation_weight=validation_weight,
                                      monitor_path=monitor_path)
        clf.train(train_loader, valid_loader)
    else:
        print("Please provide proper argument")
        return


if __name__ == '__main__':
    main()
