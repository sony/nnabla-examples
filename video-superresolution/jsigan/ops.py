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

from collections import namedtuple
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.initializer as I
import numpy as np
from utils import depth_to_space


def box_filter(x, szf):
    """
    Box filter
    """
    y = F.identity(x)
    szy = list(y.shape)
    b_filt = nn.Variable((szf, szf, 1, 1))
    b_filt.data.fill(1.)
    b_filt = b_filt / (szf ** 2)
    # 5,5,1,1
    b_filt = F.tile(b_filt, [1, 1, szy[3], 1])
    b_filt = F.transpose(b_filt, (3, 2, 0, 1))
    b_filt = F.reshape(b_filt, (6, 5, 5))
    pp = int((szf - 1) / 2)
    y = F.pad(y, (0, 0, pp, pp, pp, pp, 0, 0), mode='reflect')
    y_chw = F.transpose(y, (0, 3, 1, 2))
    y_chw = F.depthwise_convolution(y_chw, b_filt, multiplier=1, stride=(1, 1))
    y_hwc = F.transpose(y_chw, (0, 2, 3, 1))
    return y_hwc


def guided_filter(img, r, eps):
    """
    Edge preserving filter
    """
    img2 = F.concatenate(img, img * img, axis=3)
    img2 = box_filter(img2, r)
    mean = F.split(img2, axis=3)
    mean_i = F.stack(mean[0], mean[1], mean[2], axis=3)
    mean_ii = F.stack(mean[3], mean[4], mean[5], axis=3)
    var_i = mean_ii - mean_i * mean_i
    a = var_i / (var_i + eps)
    b = mean_i - a * mean_i
    ab = F.concatenate(a, b, axis=3)
    ab = box_filter(ab, r)
    mean_ab = F.split(ab, axis=3)
    mean_a = F.stack(mean_ab[0], mean_ab[1], mean_ab[2], axis=3)
    mean_b = F.stack(mean_ab[3], mean_ab[4], mean_ab[5], axis=3)
    q = mean_a * img + mean_b
    return q


def conv_2d(x, o_ch, kernel, name=None):
    """
    Convolution for JSInet
    """
    b = I.ConstantInitializer(0.)
    h = PF.convolution(x, o_ch, kernel=kernel, stride=(1, 1), pad=(1, 1), channel_last=True,
                       b_init=b, name=name)
    return h


def res_block(x, out_ch, name):
    """
    Create residual block
    """
    with nn.parameter_scope(name):
        h = conv_2d(F.relu(x), out_ch, kernel=(3, 3), name='conv/0')
        h = conv_2d(F.relu(h), out_ch, kernel=(3, 3), name='conv/1')
        h = x + h
    return h


def dyn_2d_filter(x, lf_2d, k_sz):
    """
    Dynamic 2d filtering
    """
    with nn.parameter_scope('Dynamic_2D_Filtering'):
        f_localexpand = nn.Variable.from_numpy_array(
            np.eye(k_sz[0] * k_sz[1], k_sz[0] * k_sz[1]))
        f_localexpand = F.reshape(f_localexpand,
                                  (k_sz[0], k_sz[1], 1, k_sz[0] * k_sz[1]))  # (9,9,1,81))
        f_localexpand = F.transpose(f_localexpand, (3, 0, 1, 2))  # (81,9,9,1))
        x_sz = x.shape
        x = F.reshape(x, (x_sz[0], x_sz[1], x_sz[2], 1))  # (1,100,170,1)
        x_localexpand = F.convolution(x, f_localexpand, stride=(1, 1), pad=(4, 4),
                                      channel_last=True)  # (1,100,170,81)
        x_le_sz = x_localexpand.shape
        x_localexpand = F.reshape(x_localexpand,
                                  (x_le_sz[0], x_le_sz[1], x_le_sz[2], 1, x_le_sz[3]))
        y = F.batch_matmul(x_localexpand, lf_2d)
        y_sz = y.shape
        y = F.reshape(y, (y_sz[0], y_sz[1], y_sz[2], y_sz[4]))
    return y


def dyn_2d_up_operation(x, lf_2d, k_sz, sf=2):
    """
    Dynamic 2d upsampling
    """
    with nn.parameter_scope("Dynamic_2D_Upsampling"):
        y = []
        sz = lf_2d.shape
        lf_2d_new = F.reshape(
            lf_2d, (sz[0], sz[1], sz[2], k_sz[0] * k_sz[0], sf ** 2))
        lf_2d_new = F.softmax(lf_2d_new, axis=3)
        for ch in range(3):  # loop over YUV channels
            # apply dynamic filtering operation
            temp = dyn_2d_filter(x[:, :, :, ch], lf_2d_new, k_sz)
            temp = depth_to_space(temp, sf)
            y += [temp]
        y = F.concatenate(*y, axis=3)
    return y


def dyn_sep_up_operation(x, dr_k_v, dr_k_h, k_sz, sf):
    """
    Dynamic separable upsampling operation with 1D separable local kernels.
    x: [B, H, W, C], dr_k_v: [B, H, W, 41*sf*sf], dr_k_h: [B, H, W, 41*sf*sf]
    out: [B, H*sf, W*sf, C]
    """
    sz = x.shape
    pad = k_sz // 2  # local filter pad size
    # [B, H, W, C*sf*sf]
    out_v = nn.Variable((sz[0], sz[1], sz[2], sz[3] * sf ** 2))
    out_v.data.zero()
    # [B, H, W, C*sf*sf]
    out_h = nn.Variable((sz[0], sz[1], sz[2], sz[3] * sf ** 2))
    out_h.data.zero()
    img_pad = F.pad(x, (0, 0, pad, pad, 0, 0, 0, 0))
    img_pad_y = F.reshape(img_pad[:, :, :, 0],
                          (img_pad.shape[0], img_pad.shape[1], img_pad.shape[2], 1))
    img_pad_y = F.tile(img_pad_y, [1, 1, 1, sf ** 2])
    img_pad_u = F.reshape(img_pad[:, :, :, 1],
                          (img_pad.shape[0], img_pad.shape[1], img_pad.shape[2], 1))
    img_pad_u = F.tile(img_pad_u, [1, 1, 1, sf ** 2])
    img_pad_v = F.reshape(img_pad[:, :, :, 2],
                          (img_pad.shape[0], img_pad.shape[1], img_pad.shape[2], 1))
    img_pad_v = F.tile(img_pad_v, [1, 1, 1, sf ** 2])
    img_pad = F.concatenate(img_pad_y, img_pad_u, img_pad_v, axis=3)

    # vertical 1D filter
    for i in range(k_sz):
        out_v = out_v + img_pad[:, i:i + sz[1], :, :] * F.tile(
            dr_k_v[:, :, :, i:k_sz * sf ** 2:k_sz], [1, 1, 1, 3])
    img_pad = F.pad(out_v, (0, 0, 0, 0, pad, pad, 0, 0))
    # horizontal 1D filter
    for i in range(k_sz):
        out_h = out_h + img_pad[:, :, i:i + sz[2], :] * F.tile(
            dr_k_h[:, :, :, i:k_sz * sf ** 2:k_sz], [1, 1, 1, 3])

    # depth to space upsampling (YUV)
    out = depth_to_space(out_h[:, :, :, 0:sf ** 2], sf)
    out = F.concatenate(out, depth_to_space(
        out_h[:, :, :, sf ** 2:2 * sf ** 2], sf), axis=3)
    out = F.concatenate(out, depth_to_space(
        out_h[:, :, :, 2 * sf ** 2:3 * sf ** 2], sf), axis=3)
    return out


def res_block_concat(x, out_ch, name):
    """
    Basic residual block -> [conv-relu | conv-relu] + input
    """
    with nn.parameter_scope(name):
        h = conv_2d(F.relu(x), out_ch, kernel=(3, 3), name='conv/0')
        h = conv_2d(F.relu(h), out_ch, kernel=(3, 3), name='conv/1')
        h = x[:, :, :, :out_ch] + h
    return h


def model(img, sf):
    """
    Define JSInet model
    """
    with nn.parameter_scope('Network'):
        with nn.parameter_scope('local_contrast_enhancement'):
            ## ================= Local Contrast Enhancement Subnet ============================ ##
            ch = 64
            b = guided_filter(img, 5, 0.01)
            n1 = conv_2d(b, ch, kernel=(3, 3), name='conv/0')
            for i in range(4):
                n1 = res_block(n1, ch, 'res_block/%d' % i)
            n1 = F.relu(n1)
            local_filter_2d = conv_2d(n1, (9 ** 2) * (sf ** 2), kernel=(3, 3),
                                      name='conv_k')  # [B, H, W, (9x9)*(sfxsf)]
            # dynamic 2D upsampling with 2D local filters
            pred_C = dyn_2d_up_operation(b, local_filter_2d, (9, 9), sf)
            # local contrast mask
            pred_C = 2 * F.sigmoid(pred_C)
            ## ================= Detail Restoration Subnet ============================ ##
            ch = 64
            d = F.div2(img, b + 1e-15)
        with nn.parameter_scope('detail_restoration'):
            n3 = conv_2d(d, ch, kernel=(3, 3), name='conv/0')
            for i in range(4):
                n3 = res_block(n3, ch, 'res_block/%d' % i)
                if i == 0:
                    d_feature = n3
            n3 = F.relu(n3)
            # separable 1D filters
            dr_k_h = conv_2d(n3, 41 * sf ** 2, kernel=(3, 3), name='conv_k_h')
            dr_k_v = conv_2d(n3, 41 * sf ** 2, kernel=(3, 3), name='conv_k_v')
            # dynamic separable upsampling with with separable 1D local filters
            pred_D = dyn_sep_up_operation(d, dr_k_v, dr_k_h, 41, sf)
        ## ================= Image Reconstruction Subnet ============================ ##
        with nn.parameter_scope('image_reconstruction'):
            n4 = conv_2d(img, ch, kernel=(3, 3), name='conv/0')
            for i in range(4):
                if i == 1:
                    n4 = F.concatenate(n4, d_feature, axis=3)
                    n4 = res_block_concat(n4, ch, 'res_block/%d' % i)
                else:
                    n4 = res_block(n4, ch, 'res_block/%d' % i)
            n4 = F.relu(n4)

            n4 = F.relu(conv_2d(n4, ch * sf * sf, kernel=(3, 3),
                                name='conv/1'))
            # (1,100,170,1024) -> (1,100,170,4,4,64) -> (1,100,4,170,4,64)
            # pixel shuffle
            n4 = depth_to_space(n4, sf)
            pred_I = conv_2d(n4, 3, kernel=(3, 3), name='conv/2')

    pred = F.add2(pred_I, pred_D) * pred_C
    jsinet = namedtuple('jsinet', ['pred'])
    return jsinet(pred)


def truncated_normal(w_shape, mean, std):
    """
    Numpy truncated normal
    """
    init = I.NormalInitializer()
    tmp = init(w_shape + (4,))
    valid = np.logical_and((np.less(tmp, 2)), (np.greater(tmp, -2)))
    ind = np.argmax(valid, axis=-1)
    ind1 = (np.expand_dims(ind, -1))
    trunc_norm = np.take_along_axis(tmp, ind1, axis=4).squeeze(-1)
    trunc_norm = trunc_norm * std + mean
    return trunc_norm


def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, scope='conv_0'):
    """
    Convolution for discriminator
    """
    w_n_shape = (channels, kernel, kernel, x.shape[-1])
    w_init = truncated_normal(w_n_shape, mean=0.0, std=0.02)
    b_init = I.ConstantInitializer(0.)
    with nn.parameter_scope(scope):
        if pad > 0:
            h = x.shape[1]
            if h % stride == 0:
                pad = pad * 2
            else:
                pad = max(kernel - (h % stride), 0)
            pad_top = pad // 2
            pad_bottom = pad - pad_top
            pad_left = pad // 2
            pad_right = pad - pad_left

            if pad_type == 'zero':
                x = F.pad(x, (0, 0, pad_top, pad_bottom,
                              pad_left, pad_right, 0, 0))
            if pad_type == 'reflect':
                x = F.pad(x, (0, 0, pad_top, pad_bottom, pad_left,
                              pad_right, 0, 0), mode='reflect')

        def apply_w(w):
            return PF.spectral_norm(w, dim=0)

        x = PF.convolution(x, channels, kernel=(kernel, kernel), stride=(
            stride, stride), apply_w=apply_w, w_init=w_init, b_init=b_init, with_bias=use_bias,
                           channel_last=True)

    return x


def dis_block(n, c, i, train=True):
    """
    Discriminator conv_bn_relu block
    """
    out = conv(n, channels=c, kernel=4, stride=2, pad=1, use_bias=False,
               scope='d_conv/' + str(2 * i + 2))
    out_fm = F.leaky_relu(
        PF.batch_normalization(
            out, axes=[3], batch_stat=train, name='d_bn/' + str(2 * i + 1)),
        alpha=0.2)

    out = conv(out_fm, channels=c * 2, kernel=3, stride=1, pad=1, use_bias=False,
               scope='d_conv/' + str(2 * i + 3))
    out = F.leaky_relu(
        PF.batch_normalization(
            out, axes=[3], batch_stat=train, name='d_bn/' + str(2 * i + 2)),
        alpha=0.2)
    return out, out_fm


def discriminator_fm(x, sf, scope="Discriminator_FM"):
    """
    Feature matching discriminator
    """
    with nn.parameter_scope(scope):
        fm_list = []
        ch = 32
        n = F.leaky_relu(conv(x, ch, 3, 1, 1, scope='d_conv/1'), alpha=0.2)
        for i in range(4):
            n, out_fm = dis_block(n, ch, i, train=True)
            ch = ch * 2
            fm_list.append(out_fm)
        n = F.leaky_relu(PF.batch_normalization(
            conv(n, channels=ch, kernel=4, stride=2,
                 pad=1, use_bias=False, scope='d_conv/10'),
            axes=[3], batch_stat=True, name='d_bn/9'), alpha=0.2
            )

        if sf == 1:
            n = F.leaky_relu(PF.batch_normalization(
                conv(n, channels=ch, kernel=5, stride=1,
                     pad=1, use_bias=False, scope='d_conv/11'),
                axes=[3], batch_stat=True, name='d_bn/10'), alpha=0.2)
        else:
            n = F.leaky_relu(PF.batch_normalization(
                conv(n, channels=ch, kernel=5, stride=1,
                     use_bias=False, scope='d_conv/11'),
                axes=[3], batch_stat=True, name='d_bn/10'), alpha=0.2)

        n = PF.batch_normalization(
            conv(n, channels=1, kernel=1, stride=1,
                 use_bias=False, scope='d_conv/12'),
            axes=[3], batch_stat=True, name='d_bn/11')

        out_logit = n
        out = F.sigmoid(out_logit)  # [B,1]
        return out, out_logit, fm_list


def discriminator_loss(real, fake):
    """
    Calculate discriminator loss
    """
    real_loss = F.mean(
        F.relu(1.0 - (real - F.reshape(F.mean(fake), (1, 1, 1, 1)))))
    fake_loss = F.mean(
        F.relu(1.0 + (fake - F.reshape(F.mean(real), (1, 1, 1, 1)))))
    l_d = real_loss + fake_loss
    return l_d


def generator_loss(real, fake):
    """
    Calculate generator loss
    """
    real_loss = F.mean(
        F.relu(1.0 + (real - F.reshape(F.mean(fake), (1, 1, 1, 1)))))
    fake_loss = F.mean(
        F.relu(1.0 - (fake - F.reshape(F.mean(real), (1, 1, 1, 1)))))
    l_g = real_loss + fake_loss
    return l_g


def feature_matching_loss(x, y, num=4):
    """
    Calculate feature matching loss
    """
    fm_loss = 0.0
    for i in range(num):
        fm_loss += F.mean(F.squared_error(x[i], y[i]))
    return fm_loss


def gan_model(label_ph, pred, conf):
    """
    Define GAN model with adversarial and discriminator losses and their orchestration
    """
    # Define Discriminator
    _, d_real_logits, d_real_fm_list = discriminator_fm(
        label_ph, conf.scaling_factor, scope="Discriminator_FM")
    # output of D for fake images
    _, d_fake_logits, d_fake_fm_list = discriminator_fm(
        pred, conf.scaling_factor, scope="Discriminator_FM")

    # Define Detail Discriminator
    # compute the detail layers for the dicriminator (reuse)
    base_gt = guided_filter(label_ph, 5, 0.01)
    detail_gt = F.div2(label_ph, base_gt + 1e-15)
    base_pred = guided_filter(pred, 5, 0.01)
    detail_pred = F.div2(pred, base_pred + 1e-15)

    # detail layer output of D for real images
    _, d_detail_real_logits, d_detail_real_fm_list = \
        discriminator_fm(detail_gt, conf.scaling_factor,
                         scope="Discriminator_Detail")

    # detail layer output of D for fake images
    _, d_detail_fake_logits, d_detail_fake_fm_list = \
        discriminator_fm(detail_pred, conf.scaling_factor,
                         scope="Discriminator_Detail")

    # Loss
    # original GAN (hinge GAN)
    d_adv_loss = discriminator_loss(d_real_logits, d_fake_logits)
    d_adv_loss.persistent = True
    g_adv_loss = generator_loss(d_real_logits, d_fake_logits)
    g_adv_loss.persistent = True

    # detail GAN (hinge GAN)
    d_detail_adv_loss = conf.detail_lambda * \
        discriminator_loss(d_detail_real_logits, d_detail_fake_logits)
    d_detail_adv_loss.persistent = True
    g_detail_adv_loss = conf.detail_lambda * \
        generator_loss(d_detail_real_logits, d_detail_fake_logits)
    g_detail_adv_loss.persistent = True

    # feature matching (FM) loss
    fm_loss = feature_matching_loss(d_real_fm_list, d_fake_fm_list, 4)
    fm_loss.persistent = True
    fm_detail_loss = conf.detail_lambda * feature_matching_loss(d_detail_real_fm_list,
                                                                d_detail_fake_fm_list, 4)
    fm_detail_loss.persistent = True

    jsigan = namedtuple('jsigan',
                        ['d_adv_loss', 'd_detail_adv_loss', 'g_adv_loss', 'g_detail_adv_loss',
                         'fm_loss', 'fm_detail_loss'])
    return jsigan(d_adv_loss, d_detail_adv_loss, g_adv_loss, g_detail_adv_loss, fm_loss,
                  fm_detail_loss)
