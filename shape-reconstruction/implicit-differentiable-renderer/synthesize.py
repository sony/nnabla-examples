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
from nnabla.monitor import Monitor, MonitorImage

from tqdm import tqdm

from dataset import DTUMVSDataSource
from helper import generate_raydir_camloc, generate_all_pixels
from network import render as _render


def render(pose, intrinsic, mask_obj, conf):
    assert conf.height % conf.batch_height == 0, \
      f"conf.height ({conf.height}) % conf.batch_height ({conf.batch_height}) != 0"

    W, H = conf.width, conf.height
    bh = conf.batch_height

    xy = generate_all_pixels(W, H)
    xy = xy.reshape((1, H, W, 2))

    camloc = nn.Variable([1, 3])
    raydir = nn.Variable([1, bh * W, 3])

    with nn.auto_forward(False):
        color_pred = _render(camloc, raydir, conf).reshape((1, bh, W, 3))

    rimage = np.ndarray([1, H, W, 3])
    for h in tqdm(range(0, H, bh), desc="Rendering"):
        xy_h = xy[:, h:h+bh, :, :].reshape((1, bh * W, 2))
        raydir.d, camloc.d = generate_raydir_camloc(pose, intrinsic, xy_h)

        color_pred.forward(clear_buffer=True)
        rimage[0, h:h+bh, :, :] = color_pred.d.copy()

    rimage = rimage * mask_obj
    return rimage.transpose((0, 3, 1, 2))  # NCHW


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
    monitor_image = MonitorImage(
        f"Rendered image synthesis", monitor, interval=1)

    # Load model
    nn.load_parameters(args.model_load_path)

    # Render
    pose = ds.poses[conf.valid_index:conf.valid_index+1, ...]
    intrinsic = ds.intrinsics[conf.valid_index:conf.valid_index+1, ...]
    mask_obj = ds.masks[conf.valid_index:conf.valid_index+1, ...]
    image = render(pose, intrinsic, mask_obj, conf)
    monitor_image.add(conf.valid_index, image)


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
