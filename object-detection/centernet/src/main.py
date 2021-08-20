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

import _init_paths

import os
import numpy as np

from opts import opts
from models.model import create_model, load_model

from tqdm import trange

from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed
from nnabla.ext_utils import get_extension_context
from nnabla.utils.data_iterator import data_iterator
import nnabla.solvers as S
import nnabla as nn
from nnabla.utils.save import save
import nnabla.logger as logger


from utils import setup_neu
setup_neu()
from neu.misc import init_nnabla
from neu.learning_rate_scheduler import create_learning_rate_scheduler
from trains.ctdet import Trainer, CtdetLoss
from datasets.dataset_factory import get_data_source
from detectors.detector_factory import detector_factory


def main(opt):
    '''
    NNabla configuration
    '''
    # os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    type_config = 'half' if opt.mixed_precision else 'float'
    comm = init_nnabla(ext_name=opt.extension_module,
                       device_id='0', type_config=type_config)
    nn.set_auto_forward(True)
    output_folder = os.path.join(
        opt.save_dir, "tmp.monitor.{}_{}".format(opt.arch, opt.num_layers))
    monitor = Monitor(output_folder)
    monitor_loss = None
    monitor_hm_loss = None
    monitor_wh_loss = None
    monitor_off_loss = None
    monitor_acc = None
    monitor_val_loss = None
    monitor_val_hm_loss = None
    monitor_val_wh_loss = None
    monitor_val_off_loss = None
    monitor_map = None
    monitor_time = None
    Detector = detector_factory[opt.task]
    detector = Detector(opt)

    interval = 1
    if comm.rank == 0:
        monitor_loss = MonitorSeries(
            "Training Loss", monitor, interval=interval, verbose=False)
        monitor_hm_loss = MonitorSeries(
            "hm_loss", monitor, interval=interval, verbose=False)
        monitor_wh_loss = MonitorSeries(
            "wh_loss", monitor, interval=interval, verbose=False)
        monitor_off_loss = MonitorSeries(
            "off_loss", monitor, interval=interval, verbose=False)
        monitor_val_loss = MonitorSeries(
            "Validation Loss", monitor, interval=interval, verbose=False)
        monitor_val_hm_loss = MonitorSeries(
            "val_hm_loss", monitor, interval=interval, verbose=False)
        monitor_val_wh_loss = MonitorSeries(
            "val_wh_loss", monitor, interval=interval, verbose=False)
        monitor_val_off_loss = MonitorSeries(
            "val_off_loss", monitor, interval=interval, verbose=False)
        monitor_map = MonitorSeries(
            "Val mAP", monitor, interval=interval, verbose=False)
        monitor_time = MonitorTimeElapsed(
            "time", monitor, interval=1, verbose=False)
    '''
    Data Iterators
    '''
    seed = opt.seed
    rng = np.random.RandomState(seed)
    source_factory = get_data_source(opt.dataset)
    train_source = source_factory(opt, 'train', shuffle=True, rng=rng,
                                  mixed_precision=opt.mixed_precision, channel_last=opt.channel_last)
    train_loader = data_iterator(train_source,
                                 opt.batch_size,
                                 with_memory_cache=False,
                                 with_file_cache=False
                                 )
    train_loader = train_loader.slice(rng, comm.n_procs, slice_pos=comm.rank)
    val_source = source_factory(opt, 'val', shuffle=False, rng=rng,
                                mixed_precision=opt.mixed_precision, channel_last=opt.channel_last)
    val_loader = data_iterator(val_source,
                               opt.batch_size,
                               with_memory_cache=False,
                               with_file_cache=False
                               )
    logger.info('Creating model...')
    logger.info(opt.heads)
    logger.info(f"batch size per gpu: {opt.batch_size}")
    model = create_model(opt.arch, opt.heads, opt.head_conv, opt.num_layers, training=True,
                         channel_last=opt.channel_last, pretrained_model_dir=opt.pretrained_model_dir)
    if opt.checkpoint != '':
        load_model(model, opt.checkpoint, clear=True)

    start_epoch = 0
    loss_func = CtdetLoss(opt)
    lr_sched = create_learning_rate_scheduler(
        opt.train_config.learning_rate_config)
    solver = S.Adam(alpha=lr_sched.get_lr())
    trainer = Trainer(
                model, loss_func, solver, train_loader, train_source, [
                    monitor_loss, monitor_hm_loss, monitor_wh_loss, monitor_off_loss, monitor_val_loss, monitor_val_hm_loss, monitor_val_wh_loss, monitor_val_off_loss], opt, comm)

    root_dir = opt.save_dir

    checkpoint_dir = os.path.join(root_dir, output_folder, 'checkpoints')
    start_epoch = 0
    if opt.resume_from is not None:
        start_epoch = trainer.load_checkpoint(checkpoint_dir, opt.resume_from)
        logger.info('resuming from the epoch {}'.format(start_epoch))

    for epoch in range(start_epoch, opt.num_epochs):
        lr_sched.set_epoch(epoch)
        trainer.solver.set_learning_rate(lr_sched.get_lr())
        iteration = trainer.update(epoch)
        if comm.rank == 0:
            if epoch % opt.save_intervals == 0 or epoch == (opt.num_epochs-1):
                monitor_time.add(epoch)
                trainer.save_checkpoint(checkpoint_dir, epoch)

        if epoch % opt.val_intervals == 0 or epoch == (opt.num_epochs-1):
            model.training = False
            trainer.evaluate(val_loader, epoch)
            if not opt.val_calc_map:
                num_iters = val_loader.size
                pbar = trange(num_iters, desc="[Test][exp_id:{} epoch:{}/{}]".format(
                    opt.exp_id, epoch, opt.num_epochs), disable=comm.rank > 0)
                if comm.rank == 0:
                    results = {}
                    for ind in pbar:
                        img_id = val_source.images[ind]
                        img_info = val_source.coco.loadImgs(ids=[img_id])[0]
                        img_path = os.path.join(
                            val_source.img_dir, img_info['file_name'])
                        with nn.context_scope(comm.ctx_float):
                            ret = detector.run(img_path)
                        results[img_id] = ret['results']
                    val_map = val_source.run_eval(
                        results, opt.save_dir, opt.data_dir)
                    monitor_map.add(epoch, val_map)
            model.training = True
            if comm.n_procs > 1:
                # Required to prevent timeout error of allreduce
                # at the first iteration of the next epoch.
                comm.comm.barrier()


if __name__ == '__main__':
    opt = opts().init()
    main(opt)
