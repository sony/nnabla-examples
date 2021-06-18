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
import nnabla.functions as F

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

    def __call__(self, pred_hm, pred_wh, pred_reg, hm, inds, wh, reg, reg_mask, comm=None, channel_last=False):
        with nn.context_scope(comm.ctx_float):
            loss, hm_loss, wh_loss, off_loss = 0.0, 0.0, 0.0, 0.0
            eps = 1e-4
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

    def __init__(self, model, loss_func, solver, train_data_loader, train_data_source, logger, opt, comm=None,
                 N=2000, scaling_factor=2.0):

        self.model = model
        self.loss_func = loss_func
        self.logger = logger
        self.comm = comm
        self._iteration = 0
        self.data_loader = train_data_loader
        self.data_source = train_data_source
        self.solver = solver
        self.opt = opt
        train_size = opt.train_size
        max_objs = 128
        batch_size = opt.batch_size
        self.iterations_per_epoch = int(
                np.ceil(train_size / (comm.n_procs*batch_size)))
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
            self.pred_hm = nn.Variable(
                (opt.batch_size, opt.output_h, opt.output_w, opt.num_classes))
        else:
            self._img = nn.Variable(
                (opt.batch_size, channels, opt.input_h, opt.input_w))
            self._hm = nn.Variable(
                (opt.batch_size, opt.num_classes, opt.output_h, opt.output_w))
            self.pred_hm = nn.Variable(
                (opt.batch_size, opt.num_classes, opt.output_h, opt.output_w))
        self._inds = nn.Variable((opt.batch_size, max_objs))
        self._wh = nn.Variable((opt.batch_size, max_objs, 2))
        self._reg = nn.Variable((opt.batch_size, max_objs, 2))
        self._reg_mask = nn.Variable((opt.batch_size, max_objs, 1))
        self.pred_wh_map = nn.Variable(
            (opt.batch_size, 2, opt.output_h, opt.output_w))
        self.pred_reg_map = nn.Variable(
            (opt.batch_size, 2, opt.output_h, opt.output_w))

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
            self.solver.scale_grad(1./self.scale)
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

            if self.logger[0] != None:
                if self.comm.rank == 0:
                    m_total_loss += total_loss.d.item() / self.iterations_per_epoch
                    m_hm_loss += hm_loss.d.item() / self.iterations_per_epoch
                    m_wh_loss += wh_loss.d.item() / self.iterations_per_epoch
                    m_off_loss += off_loss.d.item() / self.iterations_per_epoch
                    pbar.set_description(
                        '[Train][exp_id:{}, epoch:{}/{}||loss:{:8.4f}, hm_loss:{:8.4f}, wh_loss:{:8.4f}, off_loss:{:8.4f}, lr:{:.2e}, scale:{:.2e}]'.format(
                            self.opt.exp_id, epoch, self.opt.num_epochs, total_loss.d.item(), hm_loss.d.item(),
                            wh_loss.d.item(), off_loss.d.item(), self.solver.learning_rate(),
                            self.scale))

        if self.logger[0] != None:
            if self.comm.rank == 0:
                self.logger[0].add(epoch, m_total_loss)
                self.logger[1].add(epoch, m_hm_loss)
                self.logger[2].add(epoch, m_wh_loss)
                self.logger[3].add(epoch, m_off_loss)

    def save_checkpoint(self, path, epoch):
        # path: saved_models_dir
        from neu import checkpoint_util as cu
        os.makedirs(path, exist_ok=True)
        cu.save_checkpoint(path, epoch, self.solver)

    def load_checkpont(self, path, epoch):
        from neu import checkpoint_util as cu
        checkpoint_file = os.path.join(
            path, 'checkpoint_{}.json'.format(epoch))
        cu.load_checkpoint(checkpoint_file, self.solver)

    def compute_loss(self, data):
        # Performs forward pass.
        self._img.d, self._hm.d, self._inds.d, self._wh.d, self._reg.d, self._reg_mask.d, _ = data
        pred_hm, pred_wh, pred_reg = self.model(self._img)
        loss = self.loss_func(pred_hm, pred_wh, pred_reg, self._hm, self._inds, self._wh,
                              self._reg, self._reg_mask, self.comm, channel_last=self.opt.channel_last)
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

            pbar.set_description(
                '[Validation][exp_id:{}, epoch:{}/{}||loss:{:8.4f}, hm_loss:{:8.4f}, wh_loss:{:8.4f}, off_loss:{:8.4f}]'.format(
                    self.opt.exp_id, epoch, self.opt.num_epochs, loss['loss'].d.item(
                    ), loss['hm_loss'].d.item(),
                    loss['wh_loss'].d.item(), loss['off_loss'].d.item()))
            del loss
        if self.logger[4] != None:
            if self.comm.rank == 0:
                self.logger[4].add(epoch, total_loss /
                                   val_iterations_per_epoch)
                self.logger[5].add(epoch, hm_loss / val_iterations_per_epoch)
                self.logger[6].add(epoch, wh_loss / val_iterations_per_epoch)
                self.logger[7].add(epoch, off_loss / val_iterations_per_epoch)
