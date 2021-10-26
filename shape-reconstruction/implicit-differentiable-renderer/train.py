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
import nnabla.solvers as S
import numpy as np
import nnabla_ext
from nnabla.ext_utils import get_extension_context
from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed, MonitorImage

from dataset import data_iterator_dtumvs, DTUMVSDataSource
from helper import generate_raydir_camloc, create_monitor_path
from network import idr_loss
from synthesize import render

# TODO: impl camera parameter optimization


def main(args):
    # Setting
    device_id = args.device_id
    conf = args.conf
    path = args.conf.data_path
    B = conf.batch_size
    R = conf.n_rays
    L = conf.layers
    D = conf.depth
    feature_size = conf.feature_size

    # Dataset
    ds = DTUMVSDataSource(path, R, shuffle=True)
    di = data_iterator_dtumvs(ds, B)

    camloc = nn.Variable([B, 3])
    raydir = nn.Variable([B, R, 3])
    alpha = nn.Variable.from_numpy_array(conf.alpha)
    color_gt = nn.Variable([B, R, 3])
    mask_obj = nn.Variable([B, R, 1])

    # Monitor
    interval = di.size
    monitor_path = create_monitor_path(conf.data_path, args.monitor_path)
    monitor = Monitor(monitor_path)
    monitor_loss = MonitorSeries("Training loss", monitor, interval=interval)
    monitor_mhit = MonitorSeries("Hit count", monitor, interval=1)
    monitor_color_loss = MonitorSeries(
        "Training color loss", monitor, interval=interval)
    monitor_mask_loss = MonitorSeries(
        "Training mask loss", monitor, interval=interval)
    monitor_eikonal_loss = MonitorSeries(
        "Training eikonal loss", monitor, interval=interval)
    monitor_time = MonitorTimeElapsed(
        "Training time", monitor, interval=interval)
    monitor_image = MonitorImage("Rendered image", monitor, interval=1)

    # Solver
    solver = S.Adam(conf.learning_rate)
    loss, color_loss, mask_loss, eikonal_loss, mask_hit = \
        idr_loss(camloc, raydir, alpha, color_gt, mask_obj, conf)
    solver.set_parameters(nn.get_parameters())

    # Training loop
    for i in range(conf.train_epoch):

        ds.change_sampling_idx()

        # Validate
        if i % conf.valid_epoch_interval == 0 and not args.skip_val:
            def validate(i):
                pose_ = ds.poses[conf.valid_index:conf.valid_index+1, ...]
                intrinsic_ = ds.intrinsics[conf.valid_index:conf.valid_index+1, ...]
                mask_obj_ = ds.masks[conf.valid_index:conf.valid_index+1, ...]
                image = render(pose_, intrinsic_, mask_obj_, conf)
                monitor_image.add(i, image)
                nn.save_parameters(f"{monitor_path}/model_{i:05d}.h5")
            validate(i)

        # Train
        for j in range(di.size):
            # Feed data
            color_, mask_, intrinsic_, pose_, xy_ = di.next()
            color_gt.d = color_
            mask_obj.d = mask_
            raydir_, camloc_ = generate_raydir_camloc(pose_, intrinsic_, xy_)
            raydir.d = raydir_
            camloc.d = camloc_

            # Network
            loss.forward()
            solver.zero_grad()
            loss.backward(clear_buffer=True)
            solver.update()

            # Monitor
            t = i * di.size + j
            monitor_mhit.add(t, np.sum(mask_hit.d))
            monitor_loss.add(t, loss.d)
            monitor_color_loss.add(t, color_loss.d)
            monitor_mask_loss.add(t, mask_loss.d)
            monitor_eikonal_loss.add(t, eikonal_loss.d)
            monitor_time.add(t)

        # Decay
        if i in conf.alpha_decay:
            alpha.d = alpha.d * 2.0
        if i in conf.lr_decay:
            solver.set_learning_rate(solver.learning_rate() * 0.5)

    validate(i)


if __name__ == '__main__':
    import argparse
    import shutil
    import os
    import glob
    from ruamel.yaml import YAML
    from collections import namedtuple

    parser = argparse.ArgumentParser(
        description="Implicit Differentiable Renderer Training.")
    parser.add_argument('--device-id', '-d', type=int, default="0")
    parser.add_argument('--monitor-path', type=str, default="results")
    parser.add_argument('--config', type=str,
                        default="conf/default.yaml", required=True)
    parser.add_argument('--skip-val', action="store_true")

    args = parser.parse_args()
    with open(args.config, "r") as f:
        conf = YAML(typ='safe').load(f)
        conf = namedtuple("Conf", conf)(**conf)
    args.conf = conf

    monitor_path = create_monitor_path(conf.data_path, args.monitor_path)
    os.makedirs(monitor_path, exist_ok=True)
    shutil.copy(args.config, monitor_path)
    [shutil.copy(f, monitor_path) for f in glob.glob("*.py")]

    ctx = get_extension_context('cudnn', device_id=args.device_id)
    nn.set_default_context(ctx)
    # nn.set_auto_forward(True)

    main(args)
