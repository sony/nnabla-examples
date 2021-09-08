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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import fnmatch
import os
import nnabla as nn

import _init_paths
from datasets.dataset_factory import get_data_source
from detectors.detector_factory import detector_factory
from nnabla.ext_utils import get_extension_context
from nnabla.monitor import Monitor, MonitorSeries
from nnabla.utils.data_iterator import data_iterator
from opts import opts
from tqdm import trange


def recursive_glob(rootdir=".", pattern="*"):
    """Search recursively for files matching a specified pattern.
    """

    matches = []
    for root, dirnames, filenames in os.walk(rootdir):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))
    return sorted(matches)


def test(opt):
    """ Validate opt.checkpoint if opt.checkpoint is valid file.
    Otherwise, if opt.checkpoint_dir is set, it will search all params.h5 files and validate all of them.
    The mAP of each checkpoint will be monitored and output to opt.checkpoint_dir.

    Args:
        opt: Options

    Returns:

    """
    def test_cur_checkpoint(opt):
        if opt.checkpoint == '':
            print("Please provide trained model")
            return

        Detector = detector_factory[opt.task]
        detector = Detector(opt)

        results = {}
        num_iters = val_loader.size
        pbar = trange(num_iters, desc="[Test]")
        for ind in pbar:
            img_id = val_source.images[ind]
            img_info = val_source.coco.loadImgs(ids=[img_id])[0]
            img_path = os.path.join(val_source.img_dir, img_info['file_name'])
            ret = detector.run(img_path)
            results[img_id] = ret['results']
        mAP = val_source.run_eval(results, opt.save_dir, opt.data_dir)
        del detector
        return mAP

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    if opt.extension_module != 'cpu':
        if opt.mixed_precision:
            ctx = get_extension_context(
                opt.extension_module, device_id="0", type_config="half")
        else:
            ctx = get_extension_context(opt.extension_module, device_id="0")
        nn.set_default_context(ctx)

    nn.set_auto_forward(True)
    source_factory = get_data_source(opt.dataset)
    val_source = source_factory(opt, 'val', shuffle=False)
    batch_size = 1
    val_loader = data_iterator(val_source,
                               batch_size,
                               with_memory_cache=True,
                               with_file_cache=False
                               )

    if os.path.isdir(opt.checkpoint_dir) and os.path.exists(opt.checkpoint_dir):
        dir_path = opt.checkpoint_dir
        checkpoints_to_run = recursive_glob(dir_path, "params.h5")
        monitor = Monitor(dir_path)
        monitor_map = MonitorSeries(
            "Val mAP", monitor, interval=1, verbose=False)

        for cur_file in checkpoints_to_run:
            opt.checkpoint = cur_file
            mAP = test_cur_checkpoint(opt)
            folder_name = os.path.basename(os.path.dirname(cur_file))
            # The folder name format is defined in trains/ctdet.py.
            # Format: file_name = os.path.join(path, "epoch_" + str(epoch).zfill(3))
            epoch_num = int(folder_name.replace("epoch_", ""))
            monitor_map.add(epoch_num, mAP)

    else:
        test_cur_checkpoint(opt)


if __name__ == '__main__':
    opt = opts().init()
    test(opt)
