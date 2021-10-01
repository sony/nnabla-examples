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
import sys
import numpy as np

import nnabla as nn
import nnabla.monitor as nm
from nnabla.utils.image_utils import imread

# Set path to neu
common_utils_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))
sys.path.append(common_utils_path)
from neu.yaml_wrapper import read_yaml
from neu.misc import get_current_time


class MonitorManager(object):
    """
        input_dict = {str1: Variable1, str2: Variable2, ... }
    """

    def __init__(self, key2var_dict, monitor, interval):
        super(MonitorManager, self).__init__()
        self.key2var_dict = key2var_dict
        self.monitor_dict = dict()
        for k, v in self.key2var_dict.items():
            assert isinstance(v, nn.Variable), "invalid inputs?"
            self.monitor_dict[k] = nm.MonitorSeries(
                k, monitor, interval=interval)

    def add(self, iteration):
        for k, v in self.monitor_dict.items():
            var = self.key2var_dict[k]
            self.monitor_dict[k].add(iteration, var.d.item())


def get_monitors(config, loss_flags, loss_var_dict, test=False):

    log_root_dir = config.monitor_params.monitor_path
    log_dir = os.path.join(log_root_dir, get_current_time())

    # if additional information is given, add it
    if "info" in config.monitor_params:
        info = config.monitor_params.info
        log_dir = f'{log_dir}_{info}'

    master_monitor_misc = nm.Monitor(log_dir)
    monitor_vis = nm.MonitorImage('images', master_monitor_misc,
                                  interval=1, num_images=4,
                                  normalize_method=lambda x: x)
    if test:
        # when inference, returns the visualization monitor only
        return monitor_vis

    interval = config.monitor_params.monitor_freq
    monitoring_var_dict_gen = dict()
    monitoring_var_dict_dis = dict()

    if loss_flags.use_perceptual_loss:
        monitoring_var_dict_gen.update(
            {'perceptual_loss': loss_var_dict['perceptual_loss']})

    if loss_flags.use_gan_loss:
        monitoring_var_dict_gen.update(
            {'gan_loss_gen': loss_var_dict['gan_loss_gen']})

    if loss_flags.use_gan_loss:
        monitoring_var_dict_dis.update(
            {'gan_loss_dis': loss_var_dict['gan_loss_dis']})

    if loss_flags.use_feature_matching_loss:
        monitoring_var_dict_gen.update(
            {'feature_matching_loss': loss_var_dict['feature_matching_loss']})

    if loss_flags.use_equivariance_value_loss:
        monitoring_var_dict_gen.update(
            {'equivariance_value_loss': loss_var_dict['equivariance_value_loss']})

    if loss_flags.use_equivariance_jacobian_loss:
        monitoring_var_dict_gen.update(
            {'equivariance_jacobian_loss': loss_var_dict['equivariance_jacobian_loss']})

    monitoring_var_dict_gen.update(
        {'total_loss_gen': loss_var_dict['total_loss_gen']})

    master_monitor_gen = nm.Monitor(log_dir)
    master_monitor_dis = nm.Monitor(log_dir)

    monitors_gen = MonitorManager(monitoring_var_dict_gen,
                                  master_monitor_gen, interval=interval)
    monitors_dis = MonitorManager(monitoring_var_dict_dis,
                                  master_monitor_dis, interval=interval)
    monitor_time = nm.MonitorTimeElapsed('time_training',
                                         master_monitor_misc, interval=interval)

    return monitors_gen, monitors_dis, monitor_time, monitor_vis, log_dir


def combine_images(images):
    """
                    source        drving         fake
        images: [(B, C, H, W), (B, C, H, W), (B, C, H, W)]
    """

    batch_size = images[0].shape[0]
    target_height, target_width = images[0].shape[2:]
    header = imread("imgs/header_combined.png", channel_first=True)

    out_image = np.clip(images[0], 0.0, 1.0)
    # (3, 256, 256) -> (B, 3, 256, 256)
    header = np.tile(np.expand_dims(header, 0), (batch_size, 1, 1, 1))
    # (B, 3, 256, 256) -> (B, 3, 256, 512)
    upper_images = np.concatenate([header / 255., out_image], axis=3)

    lower_images = np.concatenate([np.clip(images[1], 0.0, 1.0),
                                   np.clip(images[2], 0.0, 1.0)], axis=3)
    out_image = np.concatenate([upper_images, lower_images], axis=2)
    return out_image
