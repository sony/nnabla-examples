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
import nnabla as nn
import nnabla.functions as F
import numpy as np

# --- for Perceptual Loss ---
from I2S_utils import VGG16_custom, MobileNet_custom
from I2S_utils import get_feature_keys
from I2S_utils import GetVariablesOnGraph


def Loss(_fake_img_var, _real_img_var, _mse_lambda=1, _pl_lambda_list=[1, 1, 1, 1], _pre_trained_network=VGG16_custom):
    pl_loss = PerceptualLoss(_fake_img_var, _real_img_var,
                             _pl_lambda_list, _pre_trained_network)
    mse_loss = MSELoss(_fake_img_var, _real_img_var, _mse_lambda)
    return pl_loss + mse_loss


def PerceptualLoss(_fake_img_var, _real_img_var, _lambda_list=[1, 1, 1, 1], _pre_trained_network=VGG16_custom):
    # input
    # _fake_img_var : type=nn.Variable(), shape=(batch_size, 3, height, width)
    # _real_img_var : type=nn.Variable(), shape=_fake_img_var.shape

    # output
    # loss : type=nn.Variable(), shape=()

    # [load VGG16]
    model = _pre_trained_network()
    pl_model_var_fake = model(
        _fake_img_var, use_up_to='lastconv+relu', training=False)
    pl_model_var_real = model(
        _real_img_var, use_up_to='lastconv+relu', training=False)
    GV_class_fake = GetVariablesOnGraph(pl_model_var_fake)
    GV_class_real = GetVariablesOnGraph(pl_model_var_real)
    # [get features]
    key_list = get_feature_keys(_pre_trained_network)
    # --- fake features ---
    conv1_1_fake = GV_class_fake.variables[key_list[0]]
    conv1_2_fake = GV_class_fake.variables[key_list[1]]
    conv3_2_fake = GV_class_fake.variables[key_list[2]]
    conv4_2_fake = GV_class_fake.variables[key_list[3]]
    feature_list_fake = [conv1_1_fake,
                         conv1_2_fake, conv3_2_fake, conv4_2_fake]
    # --- real features ---
    conv1_1_real = GV_class_real.variables[key_list[0]]
    conv1_2_real = GV_class_real.variables[key_list[1]]
    conv3_2_real = GV_class_real.variables[key_list[2]]
    conv4_2_real = GV_class_real.variables[key_list[3]]
    feature_list_real = [conv1_1_real,
                         conv1_2_real, conv3_2_real, conv4_2_real]
    # [calc loss]
    loss = 0
    for i in range(len(_lambda_list)):
        feature_fake = feature_list_fake[i]
        feature_real = feature_list_real[i]
        N = np.prod(feature_fake.shape)
        loss += F.mean(F.squared_error(feature_fake,
                       feature_real)) * (_lambda_list[i]/N)
    return loss


def MSELoss(_fake_img_var, _real_img_var, _lambda=1):
    # input
    # _fake_img_var : type=nn.Variable(), shape=(batcb_size, 3, height, width)
    # _real_img_var : type=nn.Variable(), shape=_fake_img_var.shape

    # output
    # pl_loss : type=nn.Variable(), shape=()

    N = np.prod(_fake_img_var.shape)
    return F.mean(F.squared_error(_fake_img_var, _real_img_var)) * (_lambda/N)
