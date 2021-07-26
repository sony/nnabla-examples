# Copyright (c) 2021 Sony Group Corporation. All Rights Reserved.
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


def loss_function(pred, label):
    loss = F.mean(F.softmax_cross_entropy(pred, label))
    return loss


def vgg16_prediction(image, test=False, ncls=10, seed=0):
    # Preprocess
    if not test:
        image = F.image_augmentation(
            image, min_scale=0.8, max_scale=1.2, flip_lr=True, seed=seed
        )
        image.need_grad = False
    # Convolution layers
    h = PF.convolution(
        image, 64, (3, 3), pad=(1, 1), stride=(1, 1), name="block1_conv1"
    )
    h = F.relu(h)
    h = PF.convolution(h, 64, (3, 3), pad=(
        1, 1), stride=(1, 1), name="block1_conv2")
    h = F.relu(h)
    h = F.max_pooling(h, (2, 2), stride=(2, 2))

    h = PF.convolution(h, 128, (3, 3), pad=(
        1, 1), stride=(1, 1), name="block2_conv1")
    h = F.relu(h)
    h = PF.convolution(h, 128, (3, 3), pad=(
        1, 1), stride=(1, 1), name="block2_conv2")
    h = F.relu(h)
    h = F.max_pooling(h, (2, 2), stride=(2, 2))

    h = PF.convolution(h, 256, (3, 3), pad=(
        1, 1), stride=(1, 1), name="block3_conv1")
    h = F.relu(h)
    h = PF.convolution(h, 256, (3, 3), pad=(
        1, 1), stride=(1, 1), name="block3_conv2")
    h = F.relu(h)
    h = PF.convolution(h, 256, (3, 3), pad=(
        1, 1), stride=(1, 1), name="block3_conv3")
    h = F.relu(h)
    h = F.max_pooling(h, (2, 2), stride=(2, 2))

    h = PF.convolution(h, 512, (3, 3), pad=(
        1, 1), stride=(1, 1), name="block4_conv1")
    h = F.relu(h)
    h = PF.convolution(h, 512, (3, 3), pad=(
        1, 1), stride=(1, 1), name="block4_conv2")
    h = F.relu(h)
    h = PF.convolution(h, 512, (3, 3), pad=(
        1, 1), stride=(1, 1), name="block4_conv3")
    h = F.relu(h)
    h = F.max_pooling(h, (2, 2), stride=(2, 2))

    h = PF.convolution(h, 512, (3, 3), pad=(
        1, 1), stride=(1, 1), name="block5_conv1")
    h = F.relu(h)
    h = PF.convolution(h, 512, (3, 3), pad=(
        1, 1), stride=(1, 1), name="block5_conv2")
    h = F.relu(h)
    h = PF.convolution(h, 512, (3, 3), pad=(
        1, 1), stride=(1, 1), name="block5_conv3")
    h = F.relu(h)
    hidden = F.max_pooling(h, (2, 2), stride=(2, 2))

    # Fully-Connected layers

    h = PF.affine(hidden, 4096, name="fc1")
    h = F.relu(h)
    h = PF.affine(h, 4096, name="fc2")
    h = F.relu(h)
    pred = PF.affine(h, ncls, name="fc3")

    return pred, h
