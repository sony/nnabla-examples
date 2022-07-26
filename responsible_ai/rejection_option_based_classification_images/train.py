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
import nnabla as nn
from nnabla.ext_utils import get_extension_context
import args
import data_loader as di
import classifier as cl


def train():
    """
    Train the Attribute classifier
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
    validation_weight = None
    rng = np.random.RandomState(1)
    monitor_path = os.path.join(opt['model_save_path'], opt['attribute'])
    train_loader = di.data_iterator_celeba(
        opt['celeba_image_train_dir'], opt['attr_path'], batch_size,
        target_attribute=opt['attribute'], protected_attribute=opt['protected_attribute'],
        augment=True, shuffle=True, rng=rng)
    valid_loader = di.data_iterator_celeba(
        opt['celeba_image_valid_dir'], opt['attr_path'], batch_size,
        target_attribute=opt['attribute'], protected_attribute=opt['protected_attribute'],
        rng=rng)

    clf = cl.AttributeClassifier(batch_size=batch_size, total_epochs=total_epochs,
                                 learning_rate=learning_rate,
                                 max_iter=max_iter,
                                 validation_weight=validation_weight,
                                 monitor_path=monitor_path)
    clf.train(train_loader, valid_loader)


if __name__ == '__main__':
    train()
