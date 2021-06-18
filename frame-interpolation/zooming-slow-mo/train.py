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

import os
import nnabla as nn
import nnabla.functions as F
import nnabla.solvers as S
from nnabla.ext_utils import get_extension_context
from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed
from models import zooming_slo_mo_network
from utils.lr_scheduler import get_repeated_cosine_annealing_learning_rate
from utils import CommunicatorWrapper, save_checkpoint, load_checkpoint
from args import get_config
from data_loader import data_iterator


def get_nnabla_version_integer():
    from nnabla import __version__
    import re
    r = list(map(int, re.match('^(\d+)\.(\d+)\.(\d+)', __version__).groups()))
    return r[0] * 10000 + r[1] * 100 + r[2]


def main():
    """
    main - driver code to run training for Zooming SloMo
    """
    # Check NNabla version
    if get_nnabla_version_integer() < 11700:
        raise ValueError(
            'This does not work with nnabla version less than v1.17.0 since deformable_conv layer is added in v1.17.0 . Please update the nnabla version.')

    conf = get_config()
    extension_module = conf.nnabla_context.context
    ctx = get_extension_context(
        extension_module, device_id=conf.nnabla_context.device_id)
    comm = CommunicatorWrapper(ctx)
    nn.set_default_context(comm.ctx)
    print("comm rank", comm.rank)

    # change max_iter, learning_rate and cosine_period when batch-size or no. of gpu devices change.
    default_batch_size = 12
    train_scale_factor = comm.n_procs * \
        (conf.train.batch_size / default_batch_size)
    max_iter = int(conf.train.max_iter // train_scale_factor)
    learning_rate = conf.train.learning_rate * \
        (conf.train.batch_size / default_batch_size)
    cosine_period = int(conf.train.cosine_period // train_scale_factor)

    # for single-GPU training
    data_iterator_train = data_iterator(conf, shuffle=True)

    # for multi-GPU training
    if comm.n_procs > 1:
        data_iterator_train = data_iterator_train.slice(
            rng=None, num_of_slices=comm.n_procs, slice_pos=comm.rank)

    # LR-LFR data for ZoomingSloMo input
    data_lr_lfr = nn.Variable(
              (conf.train.batch_size, (conf.data.n_frames // 2) + 1,
               3, conf.data.lr_size, conf.data.lr_size))

    # HR-HFR data for ZoomingSloMo ground truth
    data_gt = nn.Variable(
               (conf.train.batch_size, conf.data.n_frames,
                3, conf.data.gt_size, conf.data.gt_size))

    if conf.train.only_slomo:
        '''
        High resolution data as input to only-Slomo network for frame interpolation,
        hence we use lesser number of frames.
        '''
        # LFR data for SloMo input,
        slomo_gt = data_gt
        input_to_slomo = slomo_gt[:, 0:conf.data.n_frames:2, :, :, :]

    # setting up monitors for logging
    monitor_path = './nnmonitor'
    monitor = Monitor(monitor_path)
    monitor_loss = MonitorSeries(
        'loss', monitor, interval=conf.train.monitor_log_freq)
    monitor_lr = MonitorSeries(
        'learning rate', monitor, interval=conf.train.monitor_log_freq)
    monitor_time = MonitorTimeElapsed(
        "training time per iteration", monitor, interval=conf.train.monitor_log_freq)

    scope_name = "ZoomingSloMo" if not conf.train.only_slomo else "SloMo"

    with nn.parameter_scope(scope_name):
        if conf.train.only_slomo:
            generated_frame = zooming_slo_mo_network(
                input_to_slomo, conf.train.only_slomo)
            diff = generated_frame - slomo_gt
        else:
            generated_frame = zooming_slo_mo_network(
                data_lr_lfr, conf.train.only_slomo)
            diff = generated_frame - data_gt

    # Charbonnier loss
    loss = F.sum((diff * diff + conf.train.eps) ** 0.5)

    # Define optimizer
    solver = S.Adam(alpha=learning_rate, beta1=conf.train.beta1,
                    beta2=conf.train.beta2)

    # Set Parameters
    with nn.parameter_scope(scope_name):
        solver.set_parameters(nn.get_parameters())

    solver_dict = {scope_name: solver}

    if comm.rank == 0:
        print("maximum iterations", max_iter)

    start_point = 0
    if conf.train.checkpoint:
        # Load optimizer/solver information and model weights from checkpoint
        print("Loading weights from checkpoint:", conf.train.checkpoint)
        with nn.parameter_scope(scope_name):
            start_point = load_checkpoint(conf.train.checkpoint, solver_dict)

    if not os.path.isdir(conf.data.output_dir):
        os.makedirs(conf.data.output_dir)

    # Training loop.
    for i in range(start_point, max_iter):
        # Get Training Data
        if conf.train.only_slomo:
            _, data_gt.d = data_iterator_train.next()
        else:
            data_lr_lfr.d, data_gt.d = data_iterator_train.next()
        l_rate = get_repeated_cosine_annealing_learning_rate(
            i, learning_rate, conf.train.eta_min, cosine_period, conf.train.cosine_num_period)

        # Update
        solver.zero_grad()
        solver.set_learning_rate(l_rate)
        loss.forward(clear_no_need_grad=True)
        if comm.n_procs > 1:
            all_reduce_callback = comm.get_all_reduce_callback()
            loss.backward(clear_buffer=True,
                          communicator_callbacks=all_reduce_callback)
        else:
            loss.backward(clear_buffer=True)
        solver.update()

        if comm.rank == 0:
            monitor_loss.add(i, loss.d.copy())
            monitor_lr.add(i, l_rate)
            monitor_time.add(i)
            if (i % conf.train.save_checkpoint_freq) == 0:
                # Save intermediate check_points
                with nn.parameter_scope(scope_name):
                    save_checkpoint(conf.data.output_dir, i, solver_dict)

    # Save final model parameters
    if comm.rank == 0:
        with nn.parameter_scope(scope_name):
            nn.save_parameters(os.path.join(
                conf.data.output_dir, "final_model.h5"))


if __name__ == "__main__":
    main()
