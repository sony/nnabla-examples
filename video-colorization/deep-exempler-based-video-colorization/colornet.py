# Copyright 2021 Sony Group Corporation
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
import nnabla.parametric_functions as PF


def conv_rl(
    x, out_ch, kernel=(
        3, 3), stride=(
            1, 1), pad=(
                1, 1), dilation=None, name=None, act=True):
    conv = PF.convolution(
        x,
        out_ch,
        kernel=kernel,
        stride=stride,
        pad=pad,
        dilation=dilation,
        name=name)
    if act:
        conv = F.relu(conv)
    return conv


def ins_norm(x):
    conv_i_norm = F.instance_normalization(
        x, gamma=None, beta=None, channel_axis=1)
    return conv_i_norm


def conv_norm_ss(
        x,
        out_ch,
        kernel=(
            1,
            1),
    stride=(
            2,
            2),
        with_bias=False,
        group=1,
        name=None):
    conv = PF.convolution(
        x,
        out_ch,
        kernel=kernel,
        stride=stride,
        with_bias=with_bias,
        group=group,
        name=name)
    return conv


def conv_up(x, out_ch, kernel=(3, 3), stride=(1, 1), pad=(1, 1), name=None):
    upsample = F.interpolate(
        x,
        scale=(
            2,
            2),
        mode='nearest',
        align_corners=False)
    conv = PF.convolution(
        upsample,
        out_ch,
        kernel=kernel,
        pad=pad,
        stride=stride,
        name=name)
    return conv


def colorvidnet(x):
    '''
    Colorization Network 
    Args:
        x : NNabla Variable 
    Returns:
        Prediction output 
    '''
    conv1_0 = F.relu(
        PF.convolution(
            x, 32, kernel=(
                3, 3), stride=(
                1, 1), pad=(
                    1, 1), name='conv1_1.0'))
    conv1_1 = conv_rl(conv1_0, 64, name='conv1_1.2')
    conv1_2 = conv_rl(conv1_1, 64, name='conv1_2')
    conv1_2norm = ins_norm(conv1_2)
    conv1_2norm_ss = conv_norm_ss(
        conv1_2norm, 64, group=64, name='conv1_2norm_ss')
    conv2_1 = conv_rl(conv1_2norm_ss, 128, name='conv2_1')
    conv2_2 = conv_rl(conv2_1, 128, name='conv2_2')
    conv2_2norm = ins_norm(conv2_2)
    conv2_2norm_ss = conv_norm_ss(
        conv2_2norm, 128, group=128, name='conv2_2norm_ss')
    conv3_1 = conv_rl(conv2_2norm_ss, 256, name='conv3_1')
    conv3_2 = conv_rl(conv3_1, 256, name='conv3_2')
    conv3_3 = conv_rl(conv3_2, 256, name='conv3_3')
    conv3_3norm = ins_norm(conv3_3)
    conv3_3norm_ss = conv_norm_ss(
        conv3_3norm, 256, group=256, name='conv3_3norm_ss')
    conv4_1 = conv_rl(conv3_3norm_ss, 512, name='conv4_1')
    conv4_2 = conv_rl(conv4_1, 512, name='conv4_2')
    conv4_3 = conv_rl(conv4_2, 512, name='conv4_3')
    conv4_3norm = ins_norm(conv4_3)
    conv5_1 = conv_rl(
        conv4_3norm, 512, pad=(
            2, 2), dilation=(
            2, 2), name='conv5_1')
    conv5_2 = conv_rl(
        conv5_1, 512, pad=(
            2, 2), dilation=(
            2, 2), name='conv5_2')
    conv5_3 = conv_rl(
        conv5_2, 512, pad=(
            2, 2), dilation=(
            2, 2), name='conv5_3')
    conv5_3norm = ins_norm(conv5_3)
    conv6_1 = conv_rl(
        conv5_3norm, 512, pad=(
            2, 2), dilation=(
            2, 2), name='conv6_1')
    conv6_2 = conv_rl(
        conv6_1, 512, pad=(
            2, 2), dilation=(
            2, 2), name='conv6_2')
    conv6_3 = conv_rl(
        conv6_2, 512, pad=(
            2, 2), dilation=(
            2, 2), name='conv6_3')
    conv6_3norm = ins_norm(conv6_3)
    conv7_1 = conv_rl(conv6_3norm, 512, name='conv7_1')
    conv7_2 = conv_rl(conv7_1, 512, name='conv7_2')
    conv7_3 = conv_rl(conv7_2, 512, name='conv7_3')
    conv7_3norm = ins_norm(conv7_3)
    conv8_1 = conv_up(conv7_3norm, 256, name='conv8_1.1')
    conv3_3_short = conv_rl(conv3_3norm, 256, name='conv3_3_short', act=False)
    conv8_1_comb = F.relu(conv8_1 + conv3_3_short)
    conv8_2 = conv_rl(conv8_1_comb, 256, name='conv8_2')
    conv8_3 = conv_rl(conv8_2, 256, name='conv8_3')
    conv8_3norm = ins_norm(conv8_3)
    conv9_1 = conv_up(conv8_3norm, 128, name='conv9_1.1')
    conv2_2_short = conv_rl(conv2_2norm, 128, name='conv2_2_short', act=False)
    conv9_1_comb = F.relu(conv9_1 + conv2_2_short)
    conv9_2 = conv_rl(conv9_1_comb, 128, name='conv9_2')
    conv9_2norm = ins_norm(conv9_2)
    conv10_1 = conv_up(conv9_2norm, 128, name='conv10_1.1')
    conv1_2_short = conv_rl(conv1_2norm, 128, name='conv1_2_short', act=False)
    conv10_1_comb = F.relu(conv10_1 + conv1_2_short)
    conv10_2 = F.leaky_relu(
        conv_rl(
            conv10_1_comb,
            128,
            name='conv10_2',
            act=False),
        alpha=0.2,
        inplace=True)
    conv10_ab = conv_rl(
        conv10_2, 2, kernel=(
            1, 1), stride=(
            1, 1), pad=None, name='conv10_ab', act=False)
    pred_ab = F.tanh(conv10_ab) * 128
    return pred_ab
