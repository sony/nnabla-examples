# Copyright 2020,2021 Sony Corporation.
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
from lpips import LPIPS


def reconstruction_loss_lpips(fake_imgs, real_imgs):
    with nn.parameter_scope("VGG"):
        lpips = LPIPS(model="vgg", params_dir="./")
        # Following the official implementation, we use F.sum here.
        loss = F.sum(lpips(fake_imgs[0], real_imgs[0]))
        loss = loss + F.sum(lpips(fake_imgs[1], real_imgs[0]))
        loss = loss + F.sum(lpips(fake_imgs[2], real_imgs[1]))

    return loss


def loss_gen(logits):
    # Hinge loss
    loss = -F.mean(logits)
    return loss


def loss_dis_fake(logits):
    # Hinge loss (following the official implementation)
    loss = F.mean(F.relu(0.2*F.rand(shape=logits.shape) + 0.8 + logits))
    return loss


def loss_dis_real(logits, rec_imgs, part, img, lmd=1.0):
    # loss = 0.0

    # Hinge loss (following the official implementation)
    loss = F.mean(F.relu(0.2*F.rand(shape=logits.shape) + 0.8 - logits))

    # Reconstruction loss for rec_img_big (reconstructed from 8x8 features of the original image)
    # Reconstruction loss for rec_img_small (reconstructed from 8x8 features of the resized image)
    # Reconstruction loss for rec_img_part (reconstructed from a part of 16x16 features of the original image)
    if lmd > 0.0:
        # Ground-truth
        img_128 = F.interpolate(img, output_size=(128, 128))
        img_256 = F.interpolate(img, output_size=(256, 256))

        img_half = F.where(F.greater_scalar(
            part[0], 0.5), img_256[:, :, :128, :], img_256[:, :, 128:, :])
        img_part = F.where(F.greater_scalar(
            part[1], 0.5), img_half[:, :, :, :128], img_half[:, :, :, 128:])

        # Integrated perceptual loss
        loss = loss + lmd * \
            reconstruction_loss_lpips(rec_imgs, [img_128, img_part])

    return loss
