# Copyright (c) 2021 Sony Corporation. All Rights Reserved.
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


def reconstruction_loss(imgA, imgB):
    return F.mean(F.abs(imgA - imgB))


def loss_gen(logits):
    # Hinge loss
    loss = -F.mean(logits)
    return loss


def loss_dis_fake(logits):
    # Hinge loss (following the official implementation)
    loss = F.mean(F.relu(0.2*F.rand(shape=logits.shape) + 0.8 + logits))
    return loss


def loss_dis_real(logits, rec_imgs, part, img):
    # Ground-truth
    img_128 = F.interpolate(img, output_size=(128, 128))
    img_256 = F.interpolate(img, output_size=(256, 256))

    loss = 0.0
    
    # Hinge loss (following the official implementation)
    loss += F.mean(F.relu(0.2*F.rand(shape=logits.shape) + 0.8 - logits))

    # Reconstruction loss for rec_img_big (reconstructed from 8x8 features of the original image)
    loss += reconstruction_loss(rec_imgs[0], img_128)

    # Reconstruction loss for rec_img_small (reconstructed from 8x8 features of the resized image)
    loss += reconstruction_loss(rec_imgs[1], img_128)

    # Reconstruction loss for rec_img_part (reconstructed from a part of 16x16 features of the original image)
    img_half = F.where(F.greater_scalar(part[0], 0.5), img_256[:,:,:128,:], img_256[:,:,128:,:])
    img_part = F.where(F.greater_scalar(part[1], 0.5), img_half[:,:,:,:128], img_half[:,:,:,128:])
    loss += reconstruction_loss(rec_imgs[2], img_part)

    return loss



