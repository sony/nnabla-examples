
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
import sys

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF

from vggnet import vgg_net
from colornet import colorvidnet
from utils_nn import gray2rgb_batch, feature_normalize


def res_block(x, out_ch, name):
    with nn.parameter_scope(name):
        residual = x
        out = F.pad(x, (1, 1, 1, 1), 'reflect')
        out = PF.convolution(
            out, out_ch, kernel=(
                3, 3), stride=(
                1, 1), name='conv1')
        out = F.instance_normalization(
            out, gamma=None, beta=None, channel_axis=1)
        out = PF.prelu(out)
        out = F.pad(out, (1, 1, 1, 1), 'reflect')
        out = PF.convolution(
            out, out_ch, kernel=(
                3, 3), stride=(
                1, 1), name='conv2')
        out = F.instance_normalization(
            out, gamma=None, beta=None, channel_axis=1)
        out += residual
        out = PF.prelu(out)
    return out


def layer2_1(x):
    pad2_1 = F.pad(x, (1, 1, 1, 1), 'reflect')
    conv2_1 = PF.convolution(
        pad2_1, 128, kernel=(
            3, 3), stride=(
            1, 1), name='layer2_1.1')
    conv2_1 = F.instance_normalization(
        conv2_1, gamma=None, beta=None, channel_axis=1)
    conv2_1 = PF.prelu(conv2_1, name='layer2_1.3')
    pad2_2 = F.pad(conv2_1, (1, 1, 1, 1), 'reflect')
    conv2_2 = PF.convolution(
        pad2_2, 64, kernel=(
            3, 3), stride=(
            2, 2), name='layer2_1.5')
    conv2_2 = F.instance_normalization(
        conv2_2, gamma=None, beta=None, channel_axis=1)
    conv2_2 = PF.prelu(conv2_2, name='layer2_1.7')
    return conv2_2


def layer3_1(x):
    pad3_1 = F.pad(x, (1, 1, 1, 1), 'reflect')
    conv3_1 = PF.convolution(
        pad3_1, 128, kernel=(
            3, 3), stride=(
            1, 1), name='layer3_1.1')
    conv3_1 = F.instance_normalization(
        conv3_1, gamma=None, beta=None, channel_axis=1)
    conv3_1 = PF.prelu(conv3_1, name='layer3_1.3')
    pad3_2 = F.pad(conv3_1, (1, 1, 1, 1), 'reflect')
    conv3_2 = PF.convolution(
        pad3_2, 64, kernel=(
            3, 3), stride=(
            1, 1), name='layer3_1.5')
    conv3_2 = F.instance_normalization(
        conv3_2, gamma=None, beta=None, channel_axis=1)
    conv3_2 = PF.prelu(conv3_2, name='layer3_1.7')
    return conv3_2


def layer4_1(x):
    pad4_1 = F.pad(x, (1, 1, 1, 1), 'reflect')
    conv4_1 = PF.convolution(
        pad4_1, 256, kernel=(
            3, 3), stride=(
            1, 1), name='layer4_1.1')
    conv4_1 = F.instance_normalization(
        conv4_1, gamma=None, beta=None, channel_axis=1)
    conv4_1 = PF.prelu(conv4_1, name='layer4_1.3')
    pad4_2 = F.pad(conv4_1, (1, 1, 1, 1), 'reflect')
    conv4_2 = PF.convolution(
        pad4_2, 64, kernel=(
            3, 3), stride=(
            1, 1), name='layer4_1.5')
    conv4_2 = F.instance_normalization(
        conv4_2, gamma=None, beta=None, channel_axis=1)
    conv4_2 = PF.prelu(conv4_2, name='layer4_1.7')
    up4_1 = F.interpolate(
        conv4_2,
        scale=(
            2,
            2),
        mode='nearest',
        align_corners=False)
    return up4_1


def layer5_1(x):
    pad5_1 = F.pad(x, (1, 1, 1, 1), 'reflect')
    conv5_1 = PF.convolution(
        pad5_1, 256, kernel=(
            3, 3), stride=(
            1, 1), name='layer5_1.1')
    conv5_1 = F.instance_normalization(
        conv5_1, gamma=None, beta=None, channel_axis=1)
    conv5_1 = PF.prelu(conv5_1, name='layer5_1.3')
    up5_1 = F.interpolate(
        conv5_1,
        scale=(
            2,
            2),
        mode='nearest',
        align_corners=False)
    pad5_2 = F.pad(up5_1, (1, 1, 1, 1), 'reflect')
    conv5_2 = PF.convolution(
        pad5_2, 64, kernel=(
            3, 3), stride=(
            1, 1), name='layer5_1.6')
    conv5_2 = F.instance_normalization(
        conv5_2, gamma=None, beta=None, channel_axis=1)
    conv5_2 = PF.prelu(conv5_2, name='layer5_1.8')
    up5_2 = F.interpolate(
        conv5_2,
        scale=(
            2,
            2),
        mode='nearest',
        align_corners=False)
    return up5_2


def layer(x, out_channel):
    res_0 = res_block(x, out_channel, name='layer.0')
    res_1 = res_block(res_0, out_channel, name='layer.1')
    res_2 = res_block(res_1, out_channel, name='layer.2')
    return res_2


def pad_replicate(x):
    start = x[:, :, 0, :]
    end = x[:, :, -1, :]
    new = F.pad(x, (1, 1, 0, 0), 'reflect')
    new[:, :, 0, :] = start
    new[:, :, -1, :] = end
    return new


def nonlocal_net(B_lab_map,
                 relu_layers,
                 temperature=0.001 * 5,
                 detach_flag=False,
                 WTA_scale_weight=1,
                 feature_noise=0):

    batch_size = B_lab_map.shape[0]
    channel = B_lab_map.shape[1]
    image_height = B_lab_map.shape[2]
    image_width = B_lab_map.shape[3]
    feature_height = int(image_height / 4)
    feature_width = int(image_width / 4)

    feature_channel = 64
    in_channels = feature_channel * 4
    inter_channels = 256

    # layer2_1
    A_feature2_1 = layer2_1(relu_layers[0])
    B_feature2_1 = layer2_1(relu_layers[4])
    # layer3_1
    A_feature3_1 = layer3_1(relu_layers[1])
    B_feature3_1 = layer3_1(relu_layers[5])
    # layer4_1
    A_feature4_1 = layer4_1(relu_layers[2])
    B_feature4_1 = layer4_1(relu_layers[6])
    # layer5_1
    A_feature5_1 = layer5_1(relu_layers[3])
    B_feature5_1 = layer5_1(relu_layers[7])

    if A_feature5_1.shape[2] != A_feature2_1.shape[2] or A_feature5_1.shape[3] != A_feature2_1.shape[3]:
        A_feature5_1 = pad_replicate(A_feature5_1)
        B_feature5_1 = pad_replicate(B_feature5_1)
    A_features = layer(
        F.concatenate(
            A_feature2_1,
            A_feature3_1,
            A_feature4_1,
            A_feature5_1,
            axis=1),
        feature_channel * 4)
    B_features = layer(
        F.concatenate(
            B_feature2_1,
            B_feature3_1,
            B_feature4_1,
            B_feature5_1,
            axis=1),
        feature_channel * 4)
    # pairwise cosine similarity
    theta = PF.convolution(
        A_features, inter_channels, kernel=(
            1, 1), stride=(
            1, 1), name='theta')
    theta_re = F.reshape(theta, (batch_size, inter_channels, -1))
    theta_re = theta_re - F.mean(theta_re, axis=2,
                                 keepdims=True)  # center the feature
    theta_norm = F.norm(
        theta_re,
        p=2,
        axis=1,
        keepdims=True) + sys.float_info.epsilon
    theta_re = F.div2(theta_re, theta_norm)
    # 2*(feature_height*feature_width)*256
    theta_permute = F.transpose(theta_re, (0, 2, 1))
    phi = PF.convolution(
        B_features, inter_channels, kernel=(
            1, 1), stride=(
            1, 1), name='phi')
    phi_re = F.reshape(phi, (batch_size, inter_channels, -1))
    # center the feature
    phi_re = phi_re - F.mean(phi_re, axis=2, keepdims=True)
    phi_norm = F.norm(phi_re, p=2, axis=1, keepdims=True) + \
        sys.float_info.epsilon
    phi_re = F.div2(phi_re, phi_norm)
    # 2*(feature_height*feature_width)*(feature_height*feature_width)
    f = F.batch_matmul(theta_permute, phi_re)

    f_shape = f.shape
    f = F.reshape(f, (1,) + f_shape)
    f_similarity = F.reshape(f, (1,) + f_shape)
    similarity_map = F.max(f_similarity, axis=3, keepdims=True)
    similarity_map = F.reshape(
        similarity_map, (batch_size, 1, feature_height, feature_width))

    # f can be negative
    # if WTA_scale_weight == 1:
    f_WTA = f

    f_WTA = f_WTA / temperature

    f_WTA_sp = f_WTA.shape
    f_WTA = F.reshape(f_WTA, (f_WTA_sp[1], f_WTA_sp[2], f_WTA_sp[3]))
    # 2*1936*1936; softmax along the horizontal line (dim=-1)
    f_div_C = F.softmax(f_WTA, axis=2)

    # downsample the reference color
    B_lab = F.average_pooling(B_lab_map, (4, 4))
    B_lab = F.reshape(B_lab, (batch_size, channel, -1))
    B_lab = F.transpose(B_lab, (0, 2, 1))  # 2*1936*channel

    # multiply the corr map with color
    y = F.batch_matmul(f_div_C, B_lab)  # 2*1936*channel
    y = F.transpose(y, (0, 2, 1))
    y = F.reshape(
        y,
        (batch_size,
         channel,
         feature_height,
         feature_width))  # 2*3*44*44
    y = F.interpolate(y, scale=(4, 4), mode='nearest', align_corners=False)
    similarity_map = F.interpolate(
        similarity_map, scale=(
            4, 4), mode='nearest', align_corners=False)

    return y, similarity_map


def warp_color(IA_l, IB_lab, features_B, feature_noise=0, temperature=0.01):
    '''
        joint_train=True, enable grad; otherwise disable grad
    '''
    # change to rgb for feature extraction
    IA_rgb_from_gray = gray2rgb_batch(IA_l)

    A_relu1_1, A_relu2_1, A_relu3_1, A_relu4_1, A_relu5_1 = vgg_net(
        IA_rgb_from_gray, pre_process=True, fix=True)
    B_relu1_1, B_relu2_1, B_relu3_1, B_relu4_1, B_relu5_1 = features_B

    # NOTE: output the feature before normalization
    features_A = [A_relu1_1, A_relu2_1, A_relu3_1, A_relu4_1, A_relu5_1]
    layers = [A_relu2_1, A_relu3_1, A_relu4_1, A_relu5_1,
              B_relu2_1, B_relu3_1, B_relu4_1, B_relu5_1]

    def feature_norm_relu(layers):
        f_norms = []
        for relu_l in layers:
            f_norm = feature_normalize(relu_l)
            f_norms.append(f_norm)
        return f_norms
    layers = feature_norm_relu(layers)
    nonlocal_BA_lab, similarity_map = nonlocal_net(
        IB_lab, layers, temperature=0.01)

    return nonlocal_BA_lab, similarity_map, features_A


def frame_colorization(IA_lab,
                       IB_lab,
                       IA_last_lab,
                       features_B,
                       joint_training=True,
                       feature_noise=0,
                       luminance_noise=0,
                       temperature=0.01):
    # change to rgb for feature extraction
    IA_l = IA_lab[:, 0:1, :, :]

    # if luminance_noise:
    nonlocal_BA_lab, similarity_map, features_A_gray = warp_color(
        IA_l, IB_lab, features_B, feature_noise, temperature=temperature)
    nonlocal_BA_ab = nonlocal_BA_lab[:, 1:3, :, :]
    color_input = F.concatenate(
        IA_l,
        nonlocal_BA_ab,
        similarity_map,
        IA_last_lab,
        axis=1)
    with nn.parameter_scope('colornet'):
        IA_ab_predict = colorvidnet(color_input)

    return IA_ab_predict, nonlocal_BA_lab, features_A_gray
