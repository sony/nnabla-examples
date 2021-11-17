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


import functools
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF


def GLU(h):
    nc = h.shape[1]
    nc = nc // 2
    return h[:, :nc] * F.sigmoid(h[:, nc:])


def Upsample(h, nmap_out, scope_name, train=True, scale=2):
    with nn.parameter_scope(scope_name):
        def sn_w(w): return PF.spectral_norm(w, dim=0)
        h = F.interpolate(h, scale=(scale, scale), mode="nearest")
        h = PF.convolution(h, nmap_out*2, (3, 3), pad=(1, 1),
                           apply_w=sn_w, with_bias=False, name="conv1")
        h = PF.batch_normalization(h, batch_stat=train)
        h = GLU(h)
    return h


def UpsampleComp(h, nmap_out, scope_name, train=True, scale=2):
    with nn.parameter_scope(scope_name):
        def sn_w(w): return PF.spectral_norm(w, dim=0)
        h = F.interpolate(h, scale=(scale, scale), mode="nearest")
        h = PF.convolution(h, nmap_out*2, (3, 3), pad=(1, 1),
                           apply_w=sn_w, with_bias=False, name="conv1")
        h = PF.batch_normalization(h, batch_stat=train)
        h = GLU(h)
        h = PF.convolution(h, nmap_out*2, (3, 3), pad=(1, 1),
                           apply_w=sn_w, with_bias=False, name="conv2")
        h = PF.batch_normalization(h, batch_stat=train)
        h = GLU(h)
    return h


def SLE(f_large, f_small, scope_name):
    with nn.parameter_scope(scope_name):
        def sn_w(w): return PF.spectral_norm(w, dim=0)
        ada_pool_size = f_small.shape[2] // 4
        h = F.average_pooling(f_small, (ada_pool_size, ada_pool_size), stride=(
            ada_pool_size, ada_pool_size))
        h = PF.convolution(
            h, f_large.shape[1], (4, 4), apply_w=sn_w, with_bias=False, name="conv1")
        # Following the official implementation, we use swish instead of LeakyReLU here.
        h = h * F.sigmoid(h)
        h = PF.convolution(
            h, f_large.shape[1], (1, 1), apply_w=sn_w, with_bias=False, name="conv2")
        h = F.sigmoid(h)
        h = f_large * h
    return h


def Generator(z, scope_name="Generator", train=True, img_size=1024, ngf=64, big=False):
    with nn.parameter_scope(scope_name):
        # Get number of channels
        nfc_multi = {4: 16, 8: 8, 16: 4, 32: 2, 64: 2,
                     128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ngf)

        def sn_w(w): return PF.spectral_norm(w, dim=0)

        # InitLayer: ConvTranspose + BN + GLU -> 4x4
        with nn.parameter_scope("init"):
            h = PF.deconvolution(z, 2*16*ngf, (4, 4),
                                 apply_w=sn_w, with_bias=False, name="deconv0")
            h = PF.batch_normalization(h, batch_stat=train, name="bn0")
            f_4 = GLU(h)

        # Calc base features
        if big:
            f_8 = UpsampleComp(f_4, nfc[8], "up4->8", train)
        else:
            f_8 = Upsample(f_4, nfc[8], "up4->8", train)

        f_16 = Upsample(f_8, nfc[16], "up8->16", train)

        if big:
            f_32 = UpsampleComp(f_16, nfc[32], "up16->32", train)
        else:
            f_32 = Upsample(f_16, nfc[32], "up16->32", train)

        # Apply SLE
        f_64 = Upsample(f_32, nfc[64], "up32->64", train)
        if big:
            f_64 = SLE(f_64, f_4, "sle4->64")
            f_128 = UpsampleComp(f_64, nfc[128], "up64->128", train)
        else:
            f_128 = Upsample(f_64, nfc[128], "up64->128", train)
        f_128 = SLE(f_128, f_8, "sle8->128")
        f_256 = Upsample(f_128, nfc[256], "up128->256")
        f_256 = SLE(f_256, f_16, "sle16->256")
        f_last = f_256
        if img_size > 256:
            if big:
                f_512 = UpsampleComp(f_256, nfc[512], "up256->512")
            else:
                f_512 = Upsample(f_256, nfc[512], "up256->512")
            f_512 = SLE(f_512, f_32, "sle32->512")
            f_last = f_512
        if img_size > 512:
            f_1024 = Upsample(f_512, nfc[1024], "up512->1024")
            f_last = f_1024

        # Conv + Tanh -> image
        img = F.tanh(PF.convolution(f_last, 3, (3, 3), pad=(1, 1),
                                    apply_w=sn_w, with_bias=False, name="conv_last"))
        img_small = F.tanh(PF.convolution(
            f_128, 3, (1, 1), apply_w=sn_w, with_bias=False, name="conv_last_small"))
    return [img, img_small]


def Generator_early(z, scope_name="Generator", train=True, img_size=1024, ngf=64, big=False):
    with nn.parameter_scope(scope_name):
        # Get number of channels
        nfc_multi = {4: 16, 8: 8, 16: 4, 32: 2, 64: 2,
                     128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ngf)

        def sn_w(w): return PF.spectral_norm(w, dim=0)

        # InitLayer: ConvTranspose + BN + GLU -> 4x4
        with nn.parameter_scope("init"):
            h = PF.deconvolution(z, 2*16*ngf, (4, 4),
                                 apply_w=sn_w, with_bias=False, name="deconv0")
            h = PF.batch_normalization(h, batch_stat=train, name="bn0")
            f_4 = GLU(h)

        # Calc base features
        if big:
            f_8 = UpsampleComp(f_4, nfc[8], "up4->8", train)
        else:
            f_8 = Upsample(f_4, nfc[8], "up4->8", train)

        f_16 = Upsample(f_8, nfc[16], "up8->16", train)

        if big:
            f_32 = UpsampleComp(f_16, nfc[32], "up16->32", train)
        else:
            f_32 = Upsample(f_16, nfc[32], "up16->32", train)

        f_64 = Upsample(f_32, nfc[64], "up32->64", train)
    return f_8, f_16, f_32, f_64


def Generator_late(f_8, f_16, f_32, f_64, scope_name="Generator", train=True, img_size=1024, ngf=64, big=False):
    with nn.parameter_scope(scope_name):
        # Get number of channels
        nfc_multi = {4: 16, 8: 8, 16: 4, 32: 2, 64: 2,
                     128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ngf)

        def sn_w(w): return PF.spectral_norm(w, dim=0)

        f_128 = Upsample(f_64, nfc[128], "up64->128", train)
        f_128 = SLE(f_128, f_8, "sle8->128")
        f_256 = Upsample(f_128, nfc[256], "up128->256")
        f_256 = SLE(f_256, f_16, "sle16->256")
        f_last = f_256
        if img_size > 256:
            if big:
                f_512 = UpsampleComp(f_256, nfc[512], "up256->512")
            else:
                f_512 = Upsample(f_256, nfc[512], "up256->512")
            f_512 = SLE(f_512, f_32, "sle32->512")
            f_last = f_512
        if img_size > 512:
            f_1024 = Upsample(f_512, nfc[1024], "up512->1024")
            f_last = f_1024

        # Conv + Tanh -> image
        img = F.tanh(PF.convolution(f_last, 3, (3, 3), pad=(1, 1),
                                    apply_w=sn_w, with_bias=False, name="conv_last"))
        img_small = F.tanh(PF.convolution(
            f_128, 3, (1, 1), apply_w=sn_w, with_bias=False, name="conv_last_small"))
    return [img, img_small]
