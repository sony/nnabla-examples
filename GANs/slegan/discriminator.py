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


import functools
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF


def GLU(h):
    nc = h.shape[1]
    nc = nc // 2
    return h[:, :nc] * F.sigmoid(h[:, nc:])


def Upsample(h, nmap_out, scope_name, scale=2):
    with nn.parameter_scope(scope_name):
        def sn_w(w): return PF.spectral_norm(w, dim=0)
        h = F.interpolate(h, scale=(scale, scale), mode="nearest")
        h = PF.convolution(h, nmap_out*2, (3, 3), pad=(1, 1),
                           apply_w=sn_w, with_bias=False, name="conv1")
        h = PF.batch_normalization(h)
        h = GLU(h)
    return h


def Downsample(h, nmap_out, scope_name):
    with nn.parameter_scope(scope_name):
        def sn_w(w): return PF.spectral_norm(w, dim=0)
        h = PF.convolution(h, nmap_out, (4, 4), stride=(
            2, 2), pad=(1, 1), apply_w=sn_w, with_bias=False)
        h = PF.batch_normalization(h)
        h = F.leaky_relu(h, 0.2)
    return h


def DownsampleComp(h, nmap_out, scope_name):
    with nn.parameter_scope(scope_name):
        def sn_w(w): return PF.spectral_norm(w, dim=0)
        # Main
        h0 = PF.convolution(h, nmap_out, (4, 4), stride=(2, 2), pad=(
            1, 1), apply_w=sn_w, with_bias=False, name="main_conv1")
        h0 = PF.batch_normalization(h0, name="bn_main1")
        h0 = F.leaky_relu(h0, 0.2)
        h0 = PF.convolution(h0, nmap_out, (3, 3), pad=(
            1, 1), apply_w=sn_w, with_bias=False, name="main_conv2")
        h0 = PF.batch_normalization(h0, name="bn_main2")
        h0 = F.leaky_relu(h0, 0.2)
        # Direct
        h1 = F.average_pooling(h, (2, 2), stride=(2, 2))
        h1 = PF.convolution(h1, nmap_out, (1, 1), apply_w=sn_w,
                            with_bias=False, name="direct_conv1")
        h1 = PF.batch_normalization(h1, name="direct_bn1")
        h1 = F.leaky_relu(h1, 0.2)
    return (h0 + h1) / 2.0


def SLE(f_large, f_small, scope_name):
    with nn.parameter_scope(scope_name):
        def sn_w(w): return PF.spectral_norm(w, dim=0)
        ada_pool_size = f_small.shape[2] // 4
        h = F.average_pooling(f_small, (ada_pool_size, ada_pool_size), stride=(
            ada_pool_size, ada_pool_size))
        h = PF.convolution(
            h, f_large.shape[1], (4, 4), apply_w=sn_w, with_bias=False, name="conv1")
        # Following the official implementation, this implementation uses swish instead of LeakyReLU here.
        h = h * F.sigmoid(h)
        h = PF.convolution(
            h, f_large.shape[1], (1, 1), apply_w=sn_w, with_bias=False, name="conv2")
        h = F.sigmoid(h)
        h = f_large * h
    return h


def SimpleDecoder(fea, scope_name):
    # Get number of channels
    nfc_multi = {4: 16, 8: 8, 16: 4, 32: 2, 64: 2,
                 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}
    nfc = {}
    for k, v in nfc_multi.items():
        nfc[k] = int(v*32)

    with nn.parameter_scope(scope_name):
        def sn_w(w): return PF.spectral_norm(w, dim=0)
        h = Upsample(fea, nfc[16], "up8->16")
        h = Upsample(h, nfc[32], "up16->32")
        h = Upsample(h, nfc[64], "up32->64")
        h = Upsample(h, nfc[128], "up64->128")
        h = PF.convolution(h, 3, (3, 3), pad=(
            1, 1), apply_w=sn_w, with_bias=False, name="conv1")
        img = F.tanh(h)

    return img


def Discriminator(img, label="real", scope_name="Discriminator", ndf=64):
    with nn.parameter_scope(scope_name):
        if type(img) is not list:
            img_small = F.interpolate(img, output_size=(128, 128))
        else:
            img_small = img[1]
            img = img[0]

        def sn_w(w): return PF.spectral_norm(w, dim=0)
        # InitLayer: -> 256x256
        with nn.parameter_scope("init"):
            h = img
            if img.shape[2] == 1024:
                h = PF.convolution(h, ndf//8, (4, 4), stride=(2, 2),
                                   pad=(1, 1), apply_w=sn_w, with_bias=False, name="conv1")
                h = F.leaky_relu(h, 0.2)
                h = PF.convolution(h, ndf//4, (4, 4), stride=(2, 2),
                                   pad=(1, 1), apply_w=sn_w, with_bias=False, name="conv2")
                h = PF.batch_normalization(h)
                h = F.leaky_relu(h, 0.2)
            elif img.shape[2] == 512:
                h = PF.convolution(h, ndf//4, (4, 4), stride=(2, 2),
                                   pad=(1, 1), apply_w=sn_w, with_bias=False, name="conv2")
                h = F.leaky_relu(h, 0.2)
            else:
                h = PF.convolution(h, ndf//4, (3, 3), pad=(1, 1),
                                   apply_w=sn_w, with_bias=False, name="conv3")
                h = F.leaky_relu(h, 0.2)

        # Calc base features
        f_256 = h
        f_128 = DownsampleComp(f_256, ndf//2, "down256->128")
        f_64 = DownsampleComp(f_128, ndf*1, "down128->64")
        f_32 = DownsampleComp(f_64, ndf*2, "down64->32")

        # Apply SLE
        f_32 = SLE(f_32, f_256, "sle256->32")
        f_16 = DownsampleComp(f_32, ndf*4, "down32->16")
        f_16 = SLE(f_16, f_128, "sle128->16")
        f_8 = DownsampleComp(f_16, ndf*16, "down16->8")
        f_8 = SLE(f_8, f_64, "sle64->8")

        # Conv + BN + LeakyRely + Conv -> logits (5x5)
        with nn.parameter_scope("last"):
            h = PF.convolution(f_8, ndf*16, (1, 1),
                               apply_w=sn_w, with_bias=False, name="conv1")
            h = PF.batch_normalization(h)
            h = F.leaky_relu(h, 0.2)
            logit_large = PF.convolution(
                h, 1, (4, 4), apply_w=sn_w, with_bias=False, name="conv2")

        # Another path: "down_from_small" in the official code
        with nn.parameter_scope("down_from_small"):
            h_s = PF.convolution(img_small, ndf//2, (4, 4), stride=(2, 2),
                                 pad=(1, 1), apply_w=sn_w, with_bias=False, name="conv1")
            h_s = F.leaky_relu(h_s, 0.2)
            h_s = Downsample(h_s, ndf*1, "dfs64->32")
            h_s = Downsample(h_s, ndf*2, "dfs32->16")
            h_s = Downsample(h_s, ndf*4, "dfs16->8")
            fea_dec_small = h_s
            logit_small = PF.convolution(
                h_s, 1, (4, 4), apply_w=sn_w, with_bias=False, name="conv2")

        # Concatenate logits
        logits = F.concatenate(logit_large, logit_small, axis=1)

        # Reconstruct images
        rec_img_big = SimpleDecoder(f_8, "dec_big")
        rec_img_small = SimpleDecoder(fea_dec_small, "dec_small")
        part_ax2 = F.rand(shape=(img.shape[0],))
        part_ax3 = F.rand(shape=(img.shape[0],))
        f_16_ax2 = F.where(F.greater_scalar(part_ax2, 0.5),
                           f_16[:, :, :8, :], f_16[:, :, 8:, :])
        f_16_part = F.where(F.greater_scalar(part_ax3, 0.5),
                            f_16_ax2[:, :, :, :8], f_16_ax2[:, :, :, 8:])
        rec_img_part = SimpleDecoder(f_16_part, "dec_part")

    if label == "real":
        return logits, [rec_img_big, rec_img_small, rec_img_part], [part_ax2, part_ax3]
    else:
        return logits
