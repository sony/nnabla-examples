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

import numpy as np
import os
import nnabla as nn
from nnabla.monitor import MonitorSeries

from models.losses import FocalLoss
from models.losses import L1Loss
from tqdm import trange

from utils import setup_neu
setup_neu()


def ceil_to_multiple(x, mul):
    '''
    Get a minimum integer >= x of a multiple of ``mul``.
    '''
    return (x + mul - 1) // mul


class CtdetLoss(object):
    def __init__(self, opt):
        self.opt = opt
        self.crit = FocalLoss()
        self.crit_reg = L1Loss()
        self.crit_wh = self.crit_reg

    def __call__(
            self, pred_hm, pred_wh, pred_reg, hm, inds, wh, reg, reg_mask,
            comm=None, channel_last=False
    ):
        with nn.context_scope(comm.ctx_float):
            loss, hm_loss, wh_loss, off_loss = 0.0, 0.0, 0.0, 0.0
            hm_loss = self.crit.forward(pred_hm, hm)
            if self.opt.wh_weight > 0:
                wh_loss = self.crit_wh.forward(
                    pred_wh, inds, wh, reg_mask, channel_last=channel_last)
            if self.opt.reg_offset and self.opt.off_weight > 0:
                off_loss = self.crit_reg.forward(
                    pred_reg, inds, reg, reg_mask, channel_last=channel_last)
            loss = self.opt.hm_weight * hm_loss + self.opt.wh_weight * \
                wh_loss + self.opt.off_weight * off_loss
        loss.persistent = True
        hm_loss.persistent = True
        wh_loss.persistent = True
        off_loss.persistent = True
        loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                      'wh_loss': wh_loss, 'off_loss': off_loss}
        return loss_stats


class Trainer(object):
    # Super class which defines common methods for full precision and mixed precision training

    def __init__(
            self, model, solver, train_data_loader, train_data_source,
            monitor, opt, comm=None, N=2000, scaling_factor=2.0
    ):

        self.model = model
        self.loss_func = CtdetLoss(opt)
        self.comm = comm
        self._iteration = 0
        self.data_loader = train_data_loader
        self.data_source = train_data_source
        self.solver = solver
        self.opt = opt
        train_size = opt.train_size
        max_objs = opt.dataset_info['max_objs']
        batch_size = opt.batch_size
        self.iterations_per_epoch = int(
            np.ceil(train_size / (comm.n_procs * batch_size)))
        self.weight_decay = opt.weight_decay

        # Mixed Precision parameters
        self.scale = opt.loss_scaling if opt.mixed_precision else 1.0
        self.N = N
        self.scaling_factor = scaling_factor
        self._counter = 0
        self._recursive_count = 0
        self._max_recursive_count = 10
        channels = 4 if opt.mixed_precision else 3
        if opt.channel_last:
            self._img = nn.Variable(
                (opt.batch_size, opt.input_h, opt.input_w, channels))
            self._hm = nn.Variable(
                (opt.batch_size, opt.output_h, opt.output_w, opt.num_classes))
        else:
            self._img = nn.Variable(
                (opt.batch_size, channels, opt.input_h, opt.input_w))
            self._hm = nn.Variable(
                (opt.batch_size, opt.num_classes, opt.output_h, opt.output_w))
        self._inds = nn.Variable((opt.batch_size, max_objs))
        self._wh = nn.Variable((opt.batch_size, max_objs, 2))
        self._reg = nn.Variable((opt.batch_size, max_objs, 2))
        self._reg_mask = nn.Variable((opt.batch_size, max_objs, 1))

        # Construct the computation graph with dummy data.
        _ = self.model(self._img)
        self.solver.set_parameters(
            nn.get_parameters(), reset=False, retain_state=True)
        self._train_monitor = {
            'loss': None, 'hm_loss': None, 'wh_loss': None, 'off_loss': None}
        self._val_monitor = {'loss': None, 'hm_loss': None,
                             'wh_loss': None, 'off_loss': None}

        interval = 1
        if comm.rank == 0:
            self._train_monitor['loss'] = MonitorSeries(
                "Training Loss", monitor, interval=interval, verbose=False)
            self._train_monitor['hm_loss'] = MonitorSeries(
                "hm_loss", monitor, interval=interval, verbose=False)
            self._train_monitor['wh_loss'] = MonitorSeries(
                "wh_loss", monitor, interval=interval, verbose=False)
            self._train_monitor['off_loss'] = MonitorSeries(
                "off_loss", monitor, interval=interval, verbose=False)

            self._val_monitor['loss'] = MonitorSeries(
                "Validation Loss", monitor, interval=interval, verbose=False)
            self._val_monitor['hm_loss'] = MonitorSeries(
                "val_hm_loss", monitor, interval=interval, verbose=False)
            self._val_monitor['wh_loss'] = MonitorSeries(
                "val_wh_loss", monitor, interval=interval, verbose=False)
            self._val_monitor['off_loss'] = MonitorSeries(
                "val_off_loss", monitor, interval=interval, verbose=False)

    def compute_gradient(self, data):
        loss = self.compute_loss(data)
        total_loss = loss['loss']
        hm_loss = loss['hm_loss']
        wh_loss = loss['wh_loss']
        off_loss = loss['off_loss']
        self.comm.all_reduce([loss['loss'].data], division=True, inplace=False)
        self.comm.all_reduce([loss['hm_loss'].data],
                             division=True, inplace=False)
        self.comm.all_reduce([loss['wh_loss'].data],
                             division=True, inplace=False)
        self.comm.all_reduce([loss['off_loss'].data],
                             division=True, inplace=False)
        self.solver.set_parameters(
            nn.get_parameters(), reset=False, retain_state=True)
        self.solver.zero_grad()
        comm_cb = None
        backward_overlaped = True
        if backward_overlaped:
            comm_cb = self.comm.get_all_reduce_callback()
        loss["loss"].backward(self.scale, clear_buffer=True,
                              communicator_callbacks=comm_cb)
        if not backward_overlaped and self.comm.n_procs > 1:
            params = [x.grad for x in nn.get_parameters().values()]
            self.comm.all_reduce(params, division=False, inplace=False)
        if self.opt.mixed_precision:
            if self.opt.use_dynamic_loss_scaling and self.solver.check_inf_or_nan_grad():
                self.scale /= self.scaling_factor
                self._counter = 0
                # recursively recompute gradient with different scales until inf or nan dissapears
                self._recursive_count += 1
                if self._recursive_count > self._max_recursive_count:
                    # raise exception if recursive count explodes
                    raise RuntimeError(
                        "Something went wrong with gradient calculations.")
                del loss, total_loss, hm_loss, wh_loss, off_loss
                return self.compute_gradient(data)
            self._recursive_count = 0
            self.solver.scale_grad(1. / self.scale)
        return total_loss, hm_loss, wh_loss, off_loss

    def update(self, epoch):
        m_total_loss = 0.0
        m_hm_loss = 0.0
        m_wh_loss = 0.0
        m_off_loss = 0.0
        pbar = trange(self.iterations_per_epoch, disable=self.comm.rank > 0)
        for i in pbar:
            data = self.data_loader.next()
            total_loss, hm_loss, wh_loss, off_loss = self.compute_gradient(
                data)
            self.solver.weight_decay(self.weight_decay)
            self.solver.update()

            if self.opt.mixed_precision and self.opt.use_dynamic_loss_scaling:
                if self._counter > self.N:
                    self.scale *= self.scaling_factor
                    self._counter = 0
                self._counter += 1

            if self.comm.rank == 0:
                m_total_loss += total_loss.d.item() / self.iterations_per_epoch
                m_hm_loss += hm_loss.d.item() / self.iterations_per_epoch
                m_wh_loss += wh_loss.d.item() / self.iterations_per_epoch
                m_off_loss += off_loss.d.item() / self.iterations_per_epoch

                pbar_text = (
                    f"[Train] epoch:{epoch}/{self.opt.num_epochs}||"
                    f"loss:{total_loss.d.item():8.4f}, "
                    f"hm_loss:{hm_loss.d.item():8.4f}, "
                    f"wh_loss:{wh_loss.d.item():8.4f}, "
                    f"off_loss:{wh_loss.d.item():8.4f}, "
                    f"lr:{self.solver.learning_rate():.2e}, "
                    f"scale:{self.scale:.2e}"
                )
                pbar.set_description(pbar_text)

        if self.comm.rank == 0:
            self._train_monitor['loss'].add(epoch, m_total_loss)
            self._train_monitor['hm_loss'].add(epoch, m_hm_loss)
            self._train_monitor['wh_loss'].add(epoch, m_wh_loss)
            self._train_monitor['off_loss'].add(epoch, m_off_loss)

    def save_checkpoint(self, path, epoch):
        # path: saved_models_dir
        from neu import checkpoint_util as cu
        os.makedirs(path, exist_ok=True)
        cu.save_checkpoint(path, epoch, self.solver)

    def load_checkpoint(self, path, epoch):
        from neu import checkpoint_util as cu
        checkpoint_file = os.path.join(
            path, 'checkpoint_{}.json'.format(epoch))
        return cu.load_checkpoint(checkpoint_file, self.solver)

    def compute_loss(self, data):
        # Performs forward pass.
        self._img.d, self._hm.d, self._inds.d, self._wh.d, self._reg.d, self._reg_mask.d, _ = data
        pred_dict = self.model(self._img)
        pred_hm = pred_dict['hm']
        pred_wh = pred_dict['wh']
        pred_reg = pred_dict['reg']
        loss = self.loss_func(
            pred_hm, pred_wh, pred_reg, self._hm, self._inds,
            self._wh, self._reg, self._reg_mask, self.comm,
            channel_last=self.opt.channel_last
        )
        return loss

    def evaluate(self, val_data_loader, epoch):
        val_iterations_per_epoch = ceil_to_multiple(
            val_data_loader.size, self.opt.batch_size)
        total_loss = 0.0
        hm_loss = 0.0
        wh_loss = 0.0
        off_loss = 0.0
        pbar = trange(val_iterations_per_epoch, disable=self.comm.rank > 0)
        for i in pbar:
            self.solver.zero_grad()
            data = val_data_loader.next()
            loss = self.compute_loss(data)
            self.comm.all_reduce([loss['loss'].data],
                                 division=True, inplace=False)
            self.comm.all_reduce([loss['hm_loss'].data],
                                 division=True, inplace=False)
            self.comm.all_reduce([loss['wh_loss'].data],
                                 division=True, inplace=False)
            self.comm.all_reduce([loss['off_loss'].data],
                                 division=True, inplace=False)
            total_loss += loss['loss'].d.item()
            hm_loss += loss['hm_loss'].d.item()
            wh_loss += loss['wh_loss'].d.item()
            off_loss += loss['off_loss'].d.item()

            pbar_text = (
                f"[Validation] epoch:{epoch}/{self.opt.num_epochs}||"
                f"loss:{loss['loss'].d.item():8.4f}, "
                f"hm_loss:{loss['hm_loss'].d.item():8.4f}, "
                f"wh_loss:{loss['wh_loss'].d.item():8.4f}, "
                f"off_loss:{loss['off_loss'].d.item():8.4f}"
            )

            pbar.set_description(pbar_text)
            del loss

        if self.comm.rank == 0:
            self._val_monitor['loss'].add(
                epoch, total_loss / val_iterations_per_epoch)
            self._val_monitor['hm_loss'].add(
                epoch, hm_loss / val_iterations_per_epoch)
            self._val_monitor['wh_loss'].add(
                epoch, wh_loss / val_iterations_per_epoch)
            self._val_monitor['off_loss'].add(
                epoch, off_loss / val_iterations_per_epoch)
