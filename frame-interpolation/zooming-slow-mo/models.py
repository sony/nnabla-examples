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

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF


def conv2d(conv_input, n, kernel, stride, pad, name='', bias=True, init_method=None, scale=0.1):
    """
    define simple 2d convlution
    """

    if init_method == 'kaiming_normal':
        std = nn.initializer.calc_normal_std_he_forward(
            conv_input.shape[1], n, kernel=(kernel, kernel))
        w_init = nn.initializer.NormalInitializer(std * scale)
        return PF.convolution(conv_input, n, kernel=(kernel, kernel), stride=(stride, stride),
                              pad=(pad, pad), with_bias=bias, w_init=w_init, name=name)
    else:
        return PF.convolution(conv_input, n, kernel=(kernel, kernel), stride=(stride, stride),
                              pad=(pad, pad), with_bias=bias, name=name)


def pcd_align(fea1, fea2):
    """
    Alignment module using Pyramid, Cascading and Deformable convolution
    with 3 pyramid levels[L1, L2, L3].
    """

    num_filters = 64
    deformable_groups = 8
    kernel_sz, stride_ln, pad_ln = 3, 1, 1

    def deform_conv(fea, offset_input, name):
        """
        deformable convolution block
        """

        with nn.parameter_scope(name):
            channels_ = deformable_groups * 3 * kernel_sz * kernel_sz
            conv_offset_mask = conv2d(offset_input, channels_, kernel_sz, stride_ln, pad_ln,
                                      bias=True,
                                      name='conv_offset_mask')
            channels = channels_ / 3
            offset = conv_offset_mask[:, :2 * channels, :, :]
            mask = F.sigmoid(
                conv_offset_mask[:, 2 * channels:3 * channels, :, :])
            deform_conv = PF.deformable_convolution(fea, num_filters, (kernel_sz, kernel_sz),
                                                    offset, mask,
                                                    deformable_group=deformable_groups,
                                                    stride=(
                                                           stride_ln, stride_ln),
                                                    pad=(pad_ln, pad_ln),
                                                    dilation=(1, 1), with_bias=True)
        return deform_conv

    y = []
    with nn.parameter_scope('pcd_align'):
        # fea1
        # L3: level 3, 1/4 spatial size
        l3_offset = F.concatenate(fea1[2], fea2[2], axis=1)
        l3_offset = F.leaky_relu(
            conv2d(l3_offset, num_filters, kernel_sz, stride_ln, pad_ln, bias=True,
                   name='l3_offset_conv1_1'))
        l3_offset = F.leaky_relu(
            conv2d(l3_offset, num_filters, kernel_sz, stride_ln, pad_ln, bias=True,
                   name='l3_offset_conv2_1'))
        l3_fea = F.leaky_relu(deform_conv(
            fea1[2], l3_offset, name='l3_dcnpack_1'))

        # L2: level 2, 1/2 spatial size
        l2_offset = F.concatenate(fea1[1], fea2[1], axis=1)
        l2_offset = F.leaky_relu(
            conv2d(l2_offset, num_filters, kernel_sz, stride_ln, pad_ln, bias=True,
                   name='l2_offset_conv1_1'))
        l3_offset = F.interpolate(l3_offset, scale=(
            2, 2), mode='linear', align_corners=False, half_pixel=True)
        l2_offset = F.leaky_relu(
            conv2d(F.concatenate(l2_offset, l3_offset * 2, axis=1), num_filters, kernel_sz,
                   stride_ln, pad_ln, bias=True,
                   name='l2_offset_conv2_1'))
        l2_offset = F.leaky_relu(
            conv2d(l2_offset, num_filters, kernel_sz, stride_ln, pad_ln, bias=True,
                   name='l2_offset_conv3_1'))
        l2_fea = deform_conv(fea1[1], l2_offset, name='l2_dcnpack_1')
        l3_fea = F.interpolate(l3_fea, scale=(
            2, 2), mode='linear', align_corners=False, half_pixel=True)
        l2_fea = F.leaky_relu(
            conv2d(F.concatenate(l2_fea, l3_fea, axis=1), num_filters, kernel_sz, stride_ln, pad_ln,
                   bias=True, name='l2_fea_conv_1'))

        # L1: level 1, original spatial size
        l1_offset = F.concatenate(fea1[0], fea2[0], axis=1)
        l1_offset = F.leaky_relu(
            conv2d(l1_offset, num_filters, kernel_sz, stride_ln, pad_ln, bias=True,
                   name='l1_offset_conv1_1'))
        l2_offset = F.interpolate(l2_offset, scale=(
            2, 2), mode='linear', align_corners=False, half_pixel=True)
        l1_offset = F.leaky_relu(
            conv2d(F.concatenate(l1_offset, l2_offset * 2, axis=1), num_filters, kernel_sz,
                   stride_ln, pad_ln, bias=True,
                   name='l1_offset_conv2_1'))
        l1_offset = F.leaky_relu(
            conv2d(l1_offset, num_filters, kernel_sz, stride_ln, pad_ln, bias=True,
                   name='l1_offset_conv3_1'))
        l1_fea = deform_conv(fea1[0], l1_offset, name='l1_dcnpack_1')
        l2_fea = F.interpolate(l2_fea, scale=(
            2, 2), mode='linear', align_corners=False, half_pixel=True)
        l1_fea = conv2d(F.concatenate(l1_fea, l2_fea, axis=1), num_filters, kernel_sz, stride_ln,
                        pad_ln, bias=True,
                        name='l1_fea_conv_1')
        y.append(l1_fea)

        # fea2
        # L3: level 3, 1/4 spatial size
        l3_offset = F.concatenate(fea2[2], fea1[2], axis=1)
        l3_offset = F.leaky_relu(
            conv2d(l3_offset, num_filters, kernel_sz, stride_ln, pad_ln, bias=True,
                   name='l3_offset_conv1_2'))
        l3_offset = F.leaky_relu(
            conv2d(l3_offset, num_filters, kernel_sz, stride_ln, pad_ln, bias=True,
                   name='l3_offset_conv2_2'))
        l3_fea = F.leaky_relu(deform_conv(
            fea2[2], l3_offset, name='l3_dcnpack_2'))

        # L2: level 2, 1/2 spatial size
        l2_offset = F.concatenate(fea2[1], fea1[1], axis=1)
        l2_offset = F.leaky_relu(
            conv2d(l2_offset, num_filters, kernel_sz, stride_ln, pad_ln, bias=True,
                   name='l2_offset_conv1_2'))
        l3_offset = F.interpolate(l3_offset, scale=(
            2, 2), mode='linear', align_corners=False, half_pixel=True)
        l2_offset = F.leaky_relu(
            conv2d(F.concatenate(l2_offset, l3_offset * 2, axis=1), num_filters, kernel_sz,
                   stride_ln, pad_ln, bias=True,
                   name='l2_offset_conv2_2'))
        l2_offset = F.leaky_relu(
            conv2d(l2_offset, num_filters, kernel_sz, stride_ln, pad_ln, bias=True,
                   name='l2_offset_conv3_2'))
        l2_fea = deform_conv(fea2[1], l2_offset, name='l2_dcnpack_2')
        l3_fea = F.interpolate(l3_fea, scale=(
            2, 2), mode='linear', align_corners=False, half_pixel=True)
        l2_fea = F.leaky_relu(
            conv2d(F.concatenate(l2_fea, l3_fea, axis=1), num_filters, kernel_sz, stride_ln, pad_ln,
                   bias=True, name='l2_fea_conv_2'))

        # L1: level 1, original spatial size
        l1_offset = F.concatenate(fea2[0], fea1[0], axis=1)
        l1_offset = F.leaky_relu(
            conv2d(l1_offset, num_filters, kernel_sz, stride_ln, pad_ln, bias=True,
                   name='l1_offset_conv1_2'))
        l2_offset = F.interpolate(l2_offset, scale=(
            2, 2), mode='linear', align_corners=False, half_pixel=True)
        l1_offset = F.leaky_relu(
            conv2d(F.concatenate(l1_offset, l2_offset * 2, axis=1), num_filters, kernel_sz,
                   stride_ln, pad_ln, bias=True,
                   name='l1_offset_conv2_2'))
        l1_offset = F.leaky_relu(
            conv2d(l1_offset, num_filters, kernel_sz, stride_ln, pad_ln, bias=True,
                   name='l1_offset_conv3_2'))
        l1_fea = deform_conv(fea2[0], l1_offset, name='l1_dcnpack_2')
        l2_fea = F.interpolate(l2_fea, scale=(
            2, 2), mode='linear', align_corners=False, half_pixel=True)
        l1_fea = conv2d(F.concatenate(l1_fea, l2_fea, axis=1), num_filters, kernel_sz, stride_ln,
                        pad_ln, bias=True,
                        name='l1_fea_conv_2')
        y.append(l1_fea)

    y = F.concatenate(*y, axis=1)
    return y


def easy_pcd(feature_p1, feature_p2, n_filt, name):
    """
    easy 3 level pyramid cascade aligning
    input: features (feature_p1, feature_p2)
    feature size: f1 = f2 = [B, N, C, H, W]
    """

    with nn.parameter_scope(name):
        # L1: level 1, original spatial size
        l1_fea = F.stack(*[feature_p1, feature_p2], axis=1)
        batch, num_frames, channels, height, width = l1_fea.shape
        l1_fea = l1_fea.reshape((-1, channels, height, width))

        # L2: level 2, 1/2 spatial size
        l2_fea = F.leaky_relu(conv2d(l1_fea, n_filt, 3, 2, 1, bias=True, name='fea_l2_conv1')
                              )
        l2_fea = F.leaky_relu(conv2d(l2_fea, n_filt, 3, 1, 1, bias=True, name='fea_l2_conv2')
                              )

        # L3: level 3, 1/4 spatial size
        l3_fea = F.leaky_relu(conv2d(l2_fea, n_filt, 3, 2, 1, bias=True, name='fea_l3_conv1')
                              )
        l3_fea = F.leaky_relu(conv2d(l3_fea, n_filt, 3, 1, 1, bias=True, name='fea_l3_conv2')
                              )

        l1_fea = F.reshape(l1_fea, (batch, num_frames, -1,
                                    height, width), inplace=False)
        l2_fea = F.reshape(l2_fea, (batch, num_frames, -1,
                                    height // 2, width // 2), inplace=False)
        l3_fea = F.reshape(l3_fea, (batch, num_frames, -1,
                                    height // 4, width // 4), inplace=False)

        fea1 = [l1_fea[:, 0, :, :, :],
                l2_fea[:, 0, :, :, :], l3_fea[:, 0, :, :, :]]
        fea2 = [l1_fea[:, 1, :, :, :],
                l2_fea[:, 1, :, :, :], l3_fea[:, 1, :, :, :]]

        aligned_fea = pcd_align(fea1, fea2)
        fusion_fea = conv2d(aligned_fea, n_filt, 1, 1,
                            0, bias=True, name='fusion')

    return fusion_fea


def conv_lstm_cell(input_tensor, cur_state, n_filt, kernel_size):
    """
    conv lstm cell definition
    """

    def split(inp):
        _, channels, _, _ = inp.shape
        channels = channels / 4
        return inp[:, :channels, :, :], inp[:, channels:2 * channels, :, :], \
            inp[:, 2 * channels:3 * channels, :, :], \
            inp[:, 3 * channels:4 * channels, :, :]

    h_cur, c_cur = cur_state
    # concatenate along channel axis
    combined = F.concatenate(*[input_tensor, h_cur], axis=1)
    combined_conv = conv2d(combined, 4 * n_filt, kernel_size, 1, kernel_size // 2,
                           name='conv_lstm_cell')
    cc_i, cc_f, cc_o, cc_g = split(combined_conv)
    act_i = F.sigmoid(cc_i)
    act_f = F.sigmoid(cc_f)
    act_o = F.sigmoid(cc_o)
    act_g = F.tanh(cc_g)
    c_next = F.add2(act_f * c_cur, act_i * act_g)
    h_next = act_o * F.tanh(c_next)
    return h_next, c_next


def deformable_conv_lstm(input_tensor, n_filt, kernel_size):
    """
    defomable convolution lstm cell definition
    """

    hidden_state_h = nn.Variable(
        (input_tensor.shape[0], n_filt, input_tensor.shape[3], input_tensor.shape[4]))
    hidden_state_c = nn.Variable(
        (input_tensor.shape[0], n_filt, input_tensor.shape[3], input_tensor.shape[4]))
    hidden_state_h.data.zero()
    hidden_state_c.data.zero()
    seq_len = input_tensor.shape[1]
    output_inner = []

    for t_idx in range(seq_len):
        in_tensor = input_tensor[:, t_idx, :, :, :]
        h_temp = easy_pcd(in_tensor, hidden_state_h, n_filt, 'pcd_h')
        c_temp = easy_pcd(in_tensor, hidden_state_c, n_filt, 'pcd_c')
        hidden_state_h, hidden_state_c = conv_lstm_cell(
            in_tensor, [h_temp, c_temp], n_filt, kernel_size)
        output_inner.append(hidden_state_h)
    layer_output = F.stack(*output_inner, axis=1)
    return layer_output


def conv_bi_lstm(x, n_filt, kernel_size):
    x_rev = x[:, ::-1, ...]
    with nn.parameter_scope('forward_net'):
        out_fwd = deformable_conv_lstm(x, n_filt, kernel_size=kernel_size)
        out_rev = deformable_conv_lstm(x_rev, n_filt, kernel_size=kernel_size)
    rev_rev = out_rev[:, ::-1, ...]
    batch, n_frames, channels, height, width = out_fwd.shape
    result = F.concatenate(out_fwd, rev_rev, axis=2)
    result = result.reshape((batch * n_frames, -1, height, width))
    result = conv2d(result, n_filt, 1, 1, 0, name='conv_1x1')
    return result.reshape((batch, -1, channels, height, width))


def pixel_shuffle(ps_input):
    """
    Apply Depth-to-Space transformation
    input : nn.Variable of shape (B, 3*2*2, H, W)
    output : nn.Variable of shape (B, 3, H*2, W*2)
    """
    batch, depth, height, width = ps_input.shape
    output = F.reshape(ps_input, (batch, depth / 4, 2,
                                  2, height, width), inplace=False)
    output = F.reshape(F.transpose(output, (0, 1, 4, 2, 5, 3)),
                       (batch, depth / 4, height * 2, width * 2), inplace=False)
    return output


def zooming_slo_mo_network(input_imgs, only_slomo, n_filt=64,
                           front_res_blocks=5, back_res_blocks=40):
    # The residual blocks
    def residual_block(res_blk_input, output_channels=64, scope='res_block'):
        """
        define a residual block here with conv + relu + conv
        """
        with nn.parameter_scope(scope):
            feats = conv2d(res_blk_input, output_channels, 3, 1, 1,
                           name='conv1', init_method='kaiming_normal', scale=0.1)
            feats = F.relu(feats)
            feats = conv2d(feats, output_channels, 3, 1, 1,
                           name='conv2', init_method='kaiming_normal', scale=0.1)
            feats = F.add2(feats, res_blk_input)
        return feats

    batch, n_frames, channels, height, width = input_imgs.shape  # n_frames: input frames

    # extract LR features
    # L1: level 1, original spatial size
    l1_fea = F.leaky_relu(
        conv2d(F.reshape(input_imgs, (-1, channels, height, width)), n_filt, 3, 1, 1,
               name='conv_first'))

    # 5 res-blocks for feature extraction
    for i in range(0, front_res_blocks, 1):
        l1_fea = residual_block(
            l1_fea, n_filt, scope='feature_extraction/%d' % (i))

    # L2: level 2, 1/2 spatial size
    l2_fea = F.leaky_relu(conv2d(l1_fea, n_filt, 3, 2, 1,
                                 name='fea_l2_conv1'))
    l2_fea = F.leaky_relu(conv2d(l2_fea, n_filt, 3, 1, 1,
                                 name='fea_l2_conv2'))

    # L3: level 3, 1/4 spatial size
    l3_fea = F.leaky_relu(conv2d(l2_fea, n_filt, 3, 2, 1,
                                 name='fea_l3_conv1'))
    l3_fea = F.leaky_relu(conv2d(l3_fea, n_filt, 3, 1, 1,
                                 name='fea_l3_conv2'))

    l1_fea = F.reshape(l1_fea, (batch, n_frames, -1,
                                height, width), inplace=False)
    l2_fea = F.reshape(l2_fea, (batch, n_frames, -1,
                                height // 2, width // 2), inplace=False)
    l3_fea = F.reshape(l3_fea, (batch, n_frames, -1,
                                height // 4, width // 4), inplace=False)

    # align using pcd
    to_lstm_fea = []
    # 0: + fea1, fusion_fea, fea2
    # 1: + ...    ...        ...  fusion_fea, fea2
    # 2: + ...    ...        ...    ...       ...   fusion_fea, fea2
    for idx in range(n_frames - 1):
        fea1 = [l1_fea[:, idx, :, :, :],
                l2_fea[:, idx, :, :, :], l3_fea[:, idx, :, :, :]]
        fea2 = [l1_fea[:, idx + 1, :, :, :],
                l2_fea[:, idx + 1, :, :, :], l3_fea[:, idx + 1, :, :, :]]
        aligned_fea = pcd_align(fea1, fea2)
        fusion_fea = conv2d(aligned_fea, n_filt, 1, 1, 0, name='fusion')
        if idx == 0:
            to_lstm_fea.append(fea1[0])
        to_lstm_fea.append(fusion_fea)
        to_lstm_fea.append(fea2[0])
    lstm_feats = F.stack(*to_lstm_fea, axis=1)

    # align using bidirectional deformable conv-lstm
    with nn.parameter_scope('conv_bi_lstm'):
        feats = conv_bi_lstm(lstm_feats, n_filt=n_filt, kernel_size=3)
    batch_size, time_slices, channels, height, width = feats.shape
    feats = feats.reshape((batch_size * time_slices, channels, height, width))

    # 40 res-blocks for reconstriction
    for i in range(0, back_res_blocks, 1):
        feats = residual_block(feats, n_filt, scope='recon_trunk/%d' % (i))

    if only_slomo:
        out = feats
    else:
        # pixel_shuffle is only for resolution enhancement (Zooming)
        out = F.leaky_relu(pixel_shuffle(conv2d(feats, n_filt * 4, 3, 1, 1, name='upconv1'))
                           )
        out = F.leaky_relu(pixel_shuffle(conv2d(out, n_filt * 4, 3, 1, 1, name='upconv2'))
                           )

    out = F.leaky_relu(conv2d(out, n_filt, 3, 1, 1,
                              name='hrconv'))

    out = conv2d(out, 3, 3, 1, 1, name='conv_last')
    _, _, hight, width = out.shape
    outs = out.reshape((batch_size, time_slices, -1, hight, width))
    return outs
