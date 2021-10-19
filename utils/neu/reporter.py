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

from __future__ import absolute_import

import os
import sys
from collections import OrderedDict, defaultdict

import nnabla as nn
import nnabla.functions as F
import nnabla.monitor as M
import numpy as np
from nnabla.monitor import tile_images
from nnabla.utils.image_utils import imsave

from .html_creator import HtmlCreator


def get_value(val, dtype=float, reduction=True):
    """
    get float value from nn.NdArray / nn.Variable / np.ndarray / float
    """

    # get NdArray from Variable
    if isinstance(val, nn.Variable):
        val = val.data

    # get value as float
    if isinstance(val, nn.NdArray):
        assert not val.clear_called

        # take average if val has more than one element
        if reduction and val.size > 1:
            with nn.auto_forward(), nn.no_grad():
                val = F.mean(val)

        v = val.get_data("r")
    elif isinstance(val, np.ndarray):
        if reduction and val.size > 1:
            val = np.mean(val)

        v = val
    else:
        assert isinstance(val, (int, float, np.generic))
        v = val

    return dtype(v)


def get_tiled_image(img, channel_last=False):
    assert len(img.shape) == 4
    assert isinstance(img, np.ndarray)

    if channel_last:
        # nnabla.monitor.tile_images requests (B, C, H, W)
        # (B, H, W, C) -> (B, C, H, W)
        img = img.transpose(0, 3, 1, 2)

    B, C, H, W = img.shape

    # create tiled image. channel last image will be returned.
    tiled_image = tile_images(img)
    _, _, Ct = tiled_image.shape
    assert C == Ct

    return tiled_image


def save_tiled_image(img, path, channel_last=False):
    """
    Save given batched images as tiled image.
    The first axis will be handled as batch.

    Args:
        img (np.ndarray):
            Images to save. The shape should be (B, C, H, W) or (B, H, W, C) depending on `channel_last`. dtype must be np.uint8.
        path (str):
            Path to save.
        channel_last (bool):
            If True, the last axis (=3) will be handled as channel.
    """
    tiled_image = get_tiled_image(img, channel_last=channel_last)

    assert tiled_image.dtype == np.uint8

    # create directory if needed
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # save
    imsave(path, tiled_image)


class MonitorWrapper(object):
    """
    Usage:
        monitor = MonitorWrapper("path/to/save", interval=10, save_time=True)

        # Currently supports three types below
        # v1: nn.Variable()
        # v2: nn.NdArray()
        # v3: np.ndarray() or float
        vars = {"name1": v1, "name2": v2, "name3", v3}

        for epoch in range(max_epoch):
            monitor(vars, epoch)
    """

    def __init__(self, save_path, interval=1, save_time=True, silent=False):
        self.monitor = M.Monitor(save_path)
        self.interval = interval
        self.silent = silent

        self.series_monitors = {}

        if save_time:
            self.monitor_time = M.MonitorTimeElapsed(
                "Epoch time", self.monitor, interval=interval, verbose=not self.silent)

    def set_series(self, name):
        """
        Set a new series name.
        """
        assert isinstance(name, str), f"name must be string but {type(name)}."
        self.series_monitors[name] = M.MonitorSeries(name, self.monitor,
                                                     interval=self.interval,
                                                     verbose=not self.silent)

    def __call__(self, name, series_val, epoch):
        assert isinstance(name, str), "name must be string."

        if name not in self.series_monitors:
            self.set_series(name)

        self.series_monitors[name].add(epoch, get_value(series_val))

        if self.monitor_time is not None:
            self.monitor_time.add(epoch)


class Reporter(object):
    def __init__(self, comm, losses, save_path=None, nimage_per_epoch=1,
                 show_interval=20, show_keys=None):
        # losses: {"loss_name": loss_Variable, ...} or list of tuple(key, value)

        self.batch_cnt = 0
        self.piter = None
        self.comm = comm
        self.save_path = save_path
        self.nimage_per_epoch = nimage_per_epoch
        self.show_interval = show_interval

        self.losses = OrderedDict(losses)  # fix loss order
        self.epoch_losses = {k: 0. for k in losses.keys()}
        self.buff = {k: nn.NdArray() for k in losses.keys()}
        self.show_keys = list(
            losses.keys()) if show_keys is None else show_keys

        is_master = comm.rank == 0
        self.monitor = MonitorWrapper(save_path) if (
            save_path is not None and is_master) else None

        self._reset_buffer()

    def set_losses(self, losses, update=True):
        # {"loss_name": loss_Variable, ...}
        assert isinstance(losses, dict)

        for key, value in losses.items():
            if key in self.losses and not update:
                continue

            self.losses[key] = value

    def _reset_buffer(self):
        # reset buff
        for loss_name, loss in self.losses.items():
            if loss is None:
                continue
            self.buff[loss_name] = nn.NdArray()
            self.buff[loss_name].zero()

        self.flushed = True

    def _save_image(self, file_name, image):
        if isinstance(image, nn.Variable):
            img = image.data.get_data("r")
        elif isinstance(image, nn.NdArray):
            img = image.get_data("r")
        else:
            assert isinstance(image, np.ndarray)
            img = image

        dir_path = os.path.join(self.save_path, "html", "imgs")

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        save_path = os.path.join(dir_path, file_name)

        img = (img - img.min()) / (img.max() - img.min())
        imsave(save_path, img)

    def _render_html(self, epoch, image_names):
        # currently dominate only supports to add dom attrs from top to bottom.
        # So we have to re-create html from scratch at each epoch in order to place results in reverse order.
        self.html = HtmlCreator(os.path.join(
            self.save_path, "html"), redirect_interval=300)

        for e in range(epoch, -1, -1):
            self.html.add_text("epoch {}".format(e))
            image_files = []
            for i in range(self.nimage_per_epoch):
                for image_name in image_names:
                    image_files.append(
                        "_".join([image_name, str(e), str(i)]) + ".png")

            self.html.add_images(image_files,
                                 [x.split("_")[0] for x in image_files])

        self.html.save()

    def start(self, progress_iter):
        self.batch_cnt = 0
        for k in self.losses.keys():
            self.epoch_losses[k] = 0.

        self.piter = progress_iter

    def _flush_losses(self):
        if self.flushed:
            return
        self.flushed = True

        desc = "[report]"

        for loss_name, loss in self.losses.items():
            if loss is None:
                continue
            self.epoch_losses[loss_name] += self.buff[loss_name].get_data("r")

            if loss_name in self.show_keys:
                desc += " {}: {:.3f}".format(loss_name,
                                             self.epoch_losses[loss_name] / self.batch_cnt)

        # show current values
        self.piter.set_description(desc)

        self._reset_buffer()

    def __call__(self):
        # update state
        self.batch_cnt += 1
        self.flushed = False

        for loss_name, loss in self.losses.items():
            if loss is None:
                continue
            self.buff[loss_name] += loss.data

        if self.batch_cnt % self.show_interval == 0:
            self._flush_losses()

    def step(self, iteration, images=None):
        # images = {"image_name": image, ...}

        self._flush_losses()

        comm_values = {k: nn.NdArray.from_numpy_array(np.asarray(x / self.batch_cnt, dtype=np.float32))
                       for k, x in self.epoch_losses.items()}

        self.comm.all_reduce(list(comm_values.values()),
                             division=True, inplace=True)

        if self.comm.rank == 0:
            if self.monitor is not None:
                for name, val in comm_values.items():
                    self.monitor(name, val, iteration)

            if images is not None:
                # write images to files.
                images_as_tuple = tuple(images.items())

                for image_name, image in images_as_tuple:
                    assert len(image) >= self.nimage_per_epoch

                    for i in range(self.nimage_per_epoch):
                        file_name = "_".join(
                            [image_name, str(iteration), str(i)]) + ".png"
                        self._save_image(file_name, image[i])

                self._render_html(iteration, tuple(images.keys()))


class AverageLogger(object):
    def __init__(self):
        self.avg_val = 0.
        self.cnt = 0

    def reset(self):
        self.avg_val = 0.
        self.cnt = 0.

    def update(self, val):
        val = get_value(val)
        oldcnt = self.cnt
        self.cnt += 1

        # Should always be true, but just in case.
        assert self.cnt > 0, "zero division"

        # take moving average
        self.avg_val = self.avg_val * oldcnt / self.cnt + val / self.cnt

    @property
    def val(self):
        return self.avg_val


class KVReporter(object):
    """
    Usage:

        import KVReporter

        comm = init_nnabla(ext_name="cudnn", device_id=args.device_id, type_config="float")
        reporter = KVReporter(comm, save_path="path/to/logdir", show_interval=20, force_persistent=True)

        x = nn.Variable(...)
        h1 = F.affine(x, c1, name="aff1")
        h2 = F.affine(h1, c2, name="aff2")
        loss = F.mean(F.squared_error(h2, t))

        solver = S.Adam()
        solver.set_paramters(nn.get_parameters())

        # show loss
        for i in range(max_iter):
            loss.forward()
            loss.backward()
            solver.zero_grad()
            solver.update()

            # KVReporter can handle nn.Variable and nn.NdArray as well as np.ndarry.
            # Using kv_mean(name, val), you can calculate moving average.
            reporter.kv_mean("loss", loss)

            # If you don't need to take moving average for the value, use kv(name, val) instead.
            reporter.kv("iterations", i)

            # get all traced values.
            # If sync=True, all values will be synced across devices via `comm`.
            # If reset=True, KVReporter resets the current average for all values as zero.
            reporter.dump(file=sys.stdout, sync=True, reset=True)

            # save values through nnabla.Monitor
            reporter.flush_monitor(i)

    """

    def __init__(self, comm=None, save_path=None,
                 monitor_silent=True, skip_kv_to_monitor=True):
        """
        Arguments:
            comm: Communicator or CommunicatorWrapper
            save_path: str. Where to save.
            show_interval: int. Interval to report.
        """
        self.batch_cnt = 0
        self.comm = comm
        self.is_synced = False

        self.name2logger = defaultdict(AverageLogger)
        self.name2val = defaultdict(float)

        self._monitor = None
        self.is_master = comm is None or comm.rank == 0
        if self.is_master:
            self._monitor = MonitorWrapper(save_path,
                                           interval=1,
                                           silent=monitor_silent) if save_path else None

        self.monitor_cnt = 0
        self.skip_kv_to_monitor = skip_kv_to_monitor

    def get_val(self, name):
        return self.name2logger[name].val

    def reset(self, names=None):
        # Do not reset monitor_cnt

        if isinstance(names, str):
            names = [names]

        for key, logger in self.name2logger.items():
            if (names is not None) and (key not in names):
                continue

            logger.reset()

        self.is_synced = True  # No need to sync initial values

    def sync_all(self, reset=True):
        if self.is_synced:
            return

        for name, logger in sorted(self.name2logger.items(), key=lambda x: x):
            # sync across all devices
            synced_val = nn.NdArray.from_numpy_array(
                np.asarray(get_value(logger.val)))
            synced_cnt = nn.NdArray.from_numpy_array(
                np.asarray(get_value(logger.cnt)))
            if self.comm is not None:
                try:
                    synced_val *= synced_cnt
                    self.comm.all_reduce(
                        [synced_val], division=False, inplace=True)
                    self.comm.all_reduce(
                        [synced_cnt], division=False, inplace=True)
                    synced_val /= synced_cnt
                except:
                    raise ValueError(
                        f"Sync error. rank: {self.comm.rank}, key: {name}")

            # update
            self.name2val[name] = get_value(synced_val.get_data("r"))

        self.is_synced = True

        if reset:
            self.reset()

    def set_key(self, key):
        """
        set key before to prevent synchronization error.
        """

        self.name2logger[key] = self.name2logger[key]

    def desc(self, reset=True, sync=True):
        if sync:
            self.sync_all(reset=reset)

        desc = "[report]"

        for name, val in sorted(self.name2val.items(), key=lambda x: x):
            desc += " {}: {:-8.3g}".format(name, val)

        return desc

    def dump(self, file=sys.stdout, reset=True, sync=True):
        if sync:
            self.sync_all(reset=reset)

        key2str = OrderedDict()
        max_key_width = 0
        max_val_width = 0
        for name, val in sorted(self.name2val.items(), key=lambda x: x):
            key2str[name] = "{:-8.3g}".format(val)
            max_key_width = max(max_key_width, len(name))
            max_val_width = max(max_val_width, len(key2str[name]))

        if len(key2str) == 0:
            return

        line = "=" * (max_key_width + max_val_width + 5)

        out = [line]
        for name, val in key2str.items():
            key_pad = " " * (max_key_width - len(name))
            out.append("{:<}{} : {:<}".format(name, key_pad, val))
        out.append(line)

        if file is not None and hasattr(file, "write"):
            file.write("\n".join(out) + "\n")
            file.flush()

    def kv(self, name, val):
        """
        For additional information. This is not dumped to monitor. 
        """
        self.name2val[name] = val

    def kv_mean(self, name, val):
        """
        update average value by `val` for the key `name`.
        """
        # take average
        logger = self.name2logger[name]
        logger.update(get_value(val))

        self.is_synced = False

    def flush_monitor(self, iter, names=None):
        """
        write kv_mean values to monitor
        """
        self.sync_all(reset=False)

        if not self.is_master or self._monitor is None:
            return

        for name, val in self.name2val.items():
            #  skip values set by kv()
            if self.skip_kv_to_monitor and name not in self.name2logger:
                continue

            if (names is not None) and (name not in names):
                continue

            self._monitor(name, val, iter)
