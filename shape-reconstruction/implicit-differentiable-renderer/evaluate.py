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
import numpy as np
from nnabla.ext_utils import get_extension_context
from nnabla.monitor import Monitor, MonitorSeries

from dataset import DTUMVSDataSource
from synthesize import render


def psnr(image_list, image_gt_list, mask_obj_list, monitor_psnrs):
    squared_error_sum = 0
    mask_sum = 0
    for i, elms in enumerate(zip(image_list, image_gt_list, mask_obj_list)):
        image, image_gt, mask_obj = elms
        image_gt = image_gt.transpose((2, 0, 1))
        mask_obj = mask_obj.transpose((2, 0, 1))

        image = (image + 1) * 127.5
        image_gt = (image_gt + 1) * 127.5
        squared_error_sum_i = np.sum(((image - image_gt) * mask_obj) ** 2)
        mask_sum_i = np.sum(mask_obj)

        squared_error_sum += squared_error_sum_i
        mask_sum += mask_sum_i

        mse_i = squared_error_sum_i / mask_sum_i
        psnr_i = 20 * np.log10(255) - 10 * np.log10(mse_i)
        monitor_psnrs.add(i, psnr_i)

    mse = squared_error_sum / mask_sum
    psnr = 20 * np.log10(255) - 10 * np.log10(mse)
    return psnr


def main(args):
    # Setting
    device_id = args.device_id
    conf = args.conf
    path = conf.data_path
    B = conf.batch_size
    R = conf.n_rays
    L = conf.layers
    D = conf.depth
    feature_size = conf.feature_size

    ctx = get_extension_context('cudnn', device_id=device_id)
    nn.set_default_context(ctx)

    # Dataset
    ds = DTUMVSDataSource(path, R, shuffle=True)

    # Monitor
    monitor_path = "/".join(args.model_load_path.split("/")[0:-1])
    monitor = Monitor(monitor_path)
    monitor_psnrs = MonitorSeries(f"PSNRs", monitor, interval=1)
    monitor_psnr = MonitorSeries(f"PSNR", monitor, interval=1)

    # Load model
    nn.load_parameters(args.model_load_path)

    # Evaluate
    image_list = []
    for pose, intrinsic, mask_obj in zip(ds.poses, ds.intrinsics, ds.masks):
        image = render(pose[np.newaxis, ...],
                       intrinsic[np.newaxis, ...],
                       mask_obj[np.newaxis, ...],
                       conf)
        image_list.append(image)

    metric = psnr(image_list, ds.images, ds.masks, monitor_psnrs)
    monitor_psnr.add(0, metric)


if __name__ == '__main__':
    import argparse
    from ruamel.yaml import YAML
    from collections import namedtuple

    parser = argparse.ArgumentParser(
        description="Implicit Differentiable Renderer Training.")
    parser.add_argument('--device-id', '-d', type=int, default="0")
    parser.add_argument('--model-load-path', type=str, required=True)
    parser.add_argument('--config', type=str,
                        default="conf/default.yaml", required=True)

    args = parser.parse_args()
    with open(args.config, "r") as f:
        conf = YAML(typ='safe').load(f)
        conf = namedtuple("Conf", conf)(**conf)
    args.conf = conf

    main(args)
