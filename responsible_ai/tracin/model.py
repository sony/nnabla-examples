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


def categorical_error(pred, label):
    # TODO: Use F.top_n_error
    """
    Compute categorical error given score vectors and labels as
    numpy.ndarray.
    """
    pred_label = pred.argmax(1)
    return (pred_label != label.flat).mean()


def resnet23_prediction(image, test=False, ncls=10, nmaps=64, act=F.relu, seed=0):
    """
    Construct ResNet 23
    """
    # Residual Unit
    def res_unit(x, scope_name, dn=False):
        C = x.shape[1]
        with nn.parameter_scope(scope_name):
            # Conv -> BN -> Nonlinear
            with nn.parameter_scope("conv1"):
                h = PF.convolution(x, C // 2, kernel=(1, 1), pad=(0, 0),
                                   with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
                h = act(h)
            # Conv -> BN -> Nonlinear
            with nn.parameter_scope("conv2"):
                h = PF.convolution(h, C // 2, kernel=(3, 3), pad=(1, 1),
                                   with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
                h = act(h)
            # Conv -> BN
            with nn.parameter_scope("conv3"):
                h = PF.convolution(h, C, kernel=(1, 1), pad=(0, 0),
                                   with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
            # Residual -> Nonlinear
            h = act(F.add2(h, x))
            # Maxpooling
            if dn:
                h = F.max_pooling(h, kernel=(2, 2), stride=(2, 2))
            return h
    # Conv -> BN -> Nonlinear
    with nn.parameter_scope("conv1"):
        # Preprocess
        if not test:
            image = F.image_augmentation(image,
                                         min_scale=0.8,
                                         max_scale=1.2,
                                         flip_lr=True,
                                         seed=seed)
            image.need_grad = False
        h = PF.convolution(image, nmaps, kernel=(3, 3),
                           pad=(1, 1), with_bias=False)
        h = PF.batch_normalization(h, batch_stat=not test)
        h = act(h)

    h = res_unit(h, "conv2", False)    # -> 32x32
    h = res_unit(h, "conv3", True)     # -> 16x16
    h = res_unit(h, "conv4", False)    # -> 16x16
    h = res_unit(h, "conv5", True)     # -> 8x8
    h = res_unit(h, "conv6", False)    # -> 8x8
    h = res_unit(h, "conv7", True)     # -> 4x4
    h = res_unit(h, "conv8", False)    # -> 4x4
    h = F.average_pooling(h, kernel=(4, 4))  # -> 1x1
    pred = PF.affine(h, ncls)

    return pred, h


def resnet56_prediction(image, test=False, ncls=10, nmaps=64, act=F.relu, seed=0):
    """
    Construct ResNet 56
    """

    channels = [16, 32, 64]

    # Residual Unit
    def res_unit(x, scope_name, c, i):
        subsampling = i == 0 and c > 16
        strides = (2, 2) if subsampling else (1, 1)

        with nn.parameter_scope(scope_name):
            # Conv -> BN -> Nonlinear
            with nn.parameter_scope("conv1"):
                h = PF.convolution(x,
                                   c,
                                   kernel=(3, 3),
                                   pad=(1, 1),
                                   stride=strides,
                                   with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
                h = act(h)
            # Conv -> BN -> Nonlinear
            with nn.parameter_scope("conv2"):
                h = PF.convolution(h,
                                   c,
                                   kernel=(3, 3),
                                   pad=(1, 1),
                                   with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)

            if subsampling:
                # Conv -> BN
                with nn.parameter_scope("conv3"):
                    x = PF.convolution(x,
                                       c,
                                       kernel=(1, 1),
                                       pad=(0, 0),
                                       stride=(2, 2),
                                       with_bias=False)
            # Residual -> Nonlinear
            h = act(F.add2(h, x))

            return h

    # Conv -> BN -> Nonlinear
    with nn.parameter_scope("conv1"):
        # Preprocess
        if not test:
            image = F.image_augmentation(image,
                                         min_scale=0.8,
                                         max_scale=1.2,
                                         flip_lr=True,
                                         seed=seed)

            image.need_grad = False
        h = PF.convolution(image,
                           channels[0],
                           kernel=(3, 3),
                           pad=(1, 1),
                           with_bias=False)
        h = PF.batch_normalization(h, batch_stat=not test)
        h = act(h)

    for c in channels:
        h = res_unit(h, f"{c}_conv2", c, 0)
        h = res_unit(h, f"{c}_conv3", c, 1)
        h = res_unit(h, f"{c}_conv4", c, 2)
        h = res_unit(h, f"{c}_conv5", c, 3)
        h = res_unit(h, f"{c}_conv6", c, 4)
        h = res_unit(h, f"{c}_conv7", c, 5)
        h = res_unit(h, f"{c}_conv8", c, 6)
        h = res_unit(h, f"{c}_conv9", c, 7)
        h = res_unit(h, f"{c}_conv10", c, 8)

    h = F.global_average_pooling(h)  # -> 1x1
    if test:
        h.need_grad = False
    pred = PF.affine(h, ncls)

    return pred, h


def loss_function(pred, label):
    loss = F.mean(F.softmax_cross_entropy(pred, label))
    return loss
