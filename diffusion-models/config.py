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

import os


def _check_args_has(args, attr):
    assert isinstance(attr, str), f"attr must be str not {type(attr)}."
    assert hasattr(args, attr), f"args must have `{attr}` as an attribute."


def _refine_if_none(val, default):
    return default if val is None else val


def _refine_image_shape(args, default_resolution):
    args.image_size = _refine_if_none(args.image_size, default_resolution)

    args.image_shape = (args.batch_size, 3, args.image_size, args.image_size)


def _refine_lr(args):
    _check_args_has(args, "image_shape")
    _check_args_has(args, "loss_scaling")

    if max(*args.image_shape[-2:]) <= 64:
        args.lr = _refine_if_none(args.lr, 1e-4)
        args.grad_clip = -1  # disable gradient clipping
    else:
        # bigger than 64
        args.lr = _refine_if_none(args.lr, 2e-5)
        args.grad_clip = 1.

    if args.loss_scaling > (1.0 + 1e-5):
        args.lr /= args.loss_scaling


def _celebahq(args):
    # default resolution is 256
    _refine_image_shape(args, 256)

    args.dataset_root_dir = os.path.join(args.data_dir, "images")


def _cifar10(args):
    # default resolution is 32
    _refine_image_shape(args, 32)

    # cifar10 data iterator will automatically download and manage dataset.
    args.dataset_root_dir = None


def _imagenet(args):
    # default resolution is 64
    _refine_image_shape(args, 64)

    if args.image_size <= 64:
        args.dataset_root_dir = os.path.join(args.data_dir, "train_64x64")
    else:
        args.dataset_root_dir = os.path.join(args.data_dir, "ilsvrc2012")


def _custum_dataset(args):
    # default resolution is 256
    _refine_image_shape(args, 256)

    _check_args_has(args, "dataset_root_dir")


def _common(args):
    _check_args_has(args, "image_shape")

    _refine_lr(args)

    B, C, H, W = args.image_shape

    if H <= 32 and W <= 32:
        args.channel_mult = _refine_if_none(args.channel_mult, (1, 2, 2, 2))
        args.base_channels = _refine_if_none(args.base_channels, 128)
        args.num_res_blocks = _refine_if_none(args.num_res_blocks, 3)
    elif H <= 64 and W <= 64:
        args.channel_mult = _refine_if_none(args.channel_mult, (1, 2, 3, 4))
        args.base_channels = _refine_if_none(args.base_channels, 128)
        args.num_res_blocks = _refine_if_none(args.num_res_blocks, 3)
    else:
        args.channel_mult = _refine_if_none(
            args.channel_mult, (1, 1, 2, 2, 4, 4))
        args.base_channels = _refine_if_none(args.base_channels, 192)
        args.num_res_blocks = _refine_if_none(args.num_res_blocks, 2)

    # todo: more larger model


def refine_args_by_dataset(args):
    _check_args_has(args, "dataset")
    _check_args_has(args, "batch_size")
    _check_args_has(args, "data_dir")

    if args.dataset == "celebahq":
        _celebahq(args)
    elif args.dataset == "cifar10":
        _cifar10(args)
    elif args.dataset.startswith("imagenet"):
        _imagenet(args)
    else:
        _custum_dataset(args)

    _common(args)


__all__ = [refine_args_by_dataset]
