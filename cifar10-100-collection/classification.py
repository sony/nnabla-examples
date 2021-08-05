# Copyright 2017,2018,2019,2020,2021 Sony Corporation.
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

import glob
import os
import re

import functools
import numpy as np

import nnabla as nn
import nnabla.functions as F
import nnabla.solvers as S
import nnabla.utils.save
from nnabla.ext_utils import get_extension_context
from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed
from nnabla.utils.communicator_util import create_communicator, single_or_rankzero


from args import get_args
from checkpoint import save_checkpoint, load_checkpoint
from cifar10_data import data_iterator_cifar10
from cifar100_data import data_iterator_cifar100
from models import resnet23_prediction, loss_function


def backward_and_all_reduce(loss, comm, with_all_reduce_callback=False):
    params = [x.grad for x in nn.get_parameters().values()]
    if with_all_reduce_callback:
        # All-reduce gradients every 2MiB parameters during backward computation
        comm_callback = comm.get_all_reduce_callback() if comm else None
        loss.backward(clear_buffer=True,
                      communicator_callbacks=comm_callback)
    else:
        loss.backward(clear_buffer=True)
        if comm:
            comm.all_reduce(params, division=False, inplace=False)


def train(args):
    """
    Multi-Device Training

    NOTE: the communicator exposes low-level interfaces

    Steps:
    * Instantiate a communicator and set parameter variables.
    * Specify contexts for computation.
    * Initialize DataIterator.
    * Construct a computation graph for training and one for validation.
    * Initialize solver and set parameter variables to that.
    * Load checkpoint to resume previous training.
    * Create monitor instances for saving and displaying training stats.
    * Training loop
      * Computate error rate for validation data (periodically)
      * Get a next minibatch.
      * Execute forwardprop
      * Set parameter gradients zero
      * Execute backprop.
      * AllReduce for gradients
      * Solver updates parameters by using gradients computed by backprop and all reduce.
      * Compute training error
    """
    # Create Communicator and Context
    comm = create_communicator(
        ignore_error=True, extension_module=args.context, type_config=args.type_config)
    if comm:
        n_devices = comm.size
        mpi_rank = comm.rank
        device_id = comm.local_rank
    else:
        n_devices = 1
        mpi_rank = 0
        device_id = args.device_id

    if args.context == 'cpu':
        import nnabla_ext.cpu
        context = nnabla_ext.cpu.context()
    else:
        import nnabla_ext.cudnn
        context = nnabla_ext.cudnn.context(device_id=device_id)
    nn.set_default_context(context)

    n_train_samples = 50000
    n_valid_samples = 10000
    bs_valid = args.batch_size
    iter_per_epoch = int(n_train_samples / args.batch_size / n_devices)

    # Model
    rng = np.random.RandomState(313)
    comm_syncbn = comm if args.sync_bn else None
    if args.net == "cifar10_resnet23":
        prediction = functools.partial(
            resnet23_prediction, rng=rng, ncls=10, nmaps=64, act=F.relu, comm=comm_syncbn)
        data_iterator = data_iterator_cifar10
    if args.net == "cifar100_resnet23":
        prediction = functools.partial(
            resnet23_prediction, rng=rng, ncls=100, nmaps=384, act=F.elu, comm=comm_syncbn)
        data_iterator = data_iterator_cifar100

    # Create training graphs
    image_train = nn.Variable((args.batch_size, 3, 32, 32))
    label_train = nn.Variable((args.batch_size, 1))
    pred_train = prediction(image_train, test=False)
    pred_train.persistent = True
    loss_train = (loss_function(pred_train, label_train) /
                  n_devices).apply(persistent=True)
    error_train = F.mean(F.top_n_error(
        pred_train, label_train, axis=1)).apply(persistent=True)
    loss_error_train = F.sink(loss_train, error_train)

    # Create validation graphs
    image_valid = nn.Variable((bs_valid, 3, 32, 32))
    label_valid = nn.Variable((bs_valid, 1))
    pred_valid = prediction(image_valid, test=True)
    error_valid = F.mean(F.top_n_error(pred_valid, label_valid, axis=1))

    # Solvers
    solver = S.Adam()
    solver.set_parameters(nn.get_parameters())
    base_lr = args.learning_rate
    warmup_iter = iter_per_epoch * args.warmup_epoch
    warmup_slope = base_lr * (n_devices - 1) / warmup_iter
    solver.set_learning_rate(base_lr)

    # load checkpoint if file exist.
    start_point = 0
    if args.use_latest_checkpoint:
        files = glob.glob(f'{args.model_save_path}/checkpoint_*.json')
        if len(files) != 0:
            index = max(
                [int(n) for n in [re.sub(r'.*checkpoint_(\d+).json', '\\1', f) for f in files]])
            # load weights and solver state info from specified checkpoint file.
            start_point = load_checkpoint(
                f'{args.model_save_path}/checkpoint_{index}.json', solver)
        print(f'checkpoint is loaded. start iteration from {start_point}')

    # Create monitor
    monitor = Monitor(args.monitor_path)
    monitor_loss = MonitorSeries("Training loss", monitor, interval=10)
    monitor_err = MonitorSeries("Training error", monitor, interval=10)
    monitor_time = MonitorTimeElapsed("Training time", monitor, interval=10)
    monitor_verr = MonitorSeries("Validation error", monitor, interval=1)
    monitor_vtime = MonitorTimeElapsed("Validation time", monitor, interval=1)

    # Data Iterator

    # If the data does not exist, it will try to download it from the server
    # and prepare it. When executing multiple processes on the same host, it is
    # necessary to execute initial data preparation by the representative
    # process (rank is 0) on the host.

    # Download dataset by rank-0 process
    if single_or_rankzero():
        rng = np.random.RandomState(mpi_rank)
        _, tdata = data_iterator(args.batch_size, True, rng)
        vsource, vdata = data_iterator(bs_valid, False)

    # Wait for data to be prepared without watchdog
    if comm:
        comm.barrier()

    # Prepare dataset for remaining process
    if not single_or_rankzero():
        rng = np.random.RandomState(mpi_rank)
        _, tdata = data_iterator(args.batch_size, True, rng)
        vsource, vdata = data_iterator(bs_valid, False)

    # Training-loop
    ve = nn.Variable()
    for i in range(start_point // n_devices, args.epochs * iter_per_epoch):
        # Validation
        if i % iter_per_epoch == 0:
            ve_local = 0.
            k = 0
            idx = np.random.permutation(n_valid_samples)
            val_images = vsource.images[idx]
            val_labels = vsource.labels[idx]
            for j in range(int(n_valid_samples / n_devices * mpi_rank),
                           int(n_valid_samples / n_devices * (mpi_rank + 1)),
                           bs_valid):
                image = val_images[j:j + bs_valid]
                label = val_labels[j:j + bs_valid]
                if len(image) != bs_valid:  # note that smaller batch is ignored
                    continue
                image_valid.d = image
                label_valid.d = label
                error_valid.forward(clear_buffer=True)
                ve_local += error_valid.d.copy()
                k += 1
            ve_local /= k
            ve.d = ve_local
            if comm:
                comm.all_reduce(ve.data, division=True, inplace=True)

            # Monitoring error and elapsed time
            if single_or_rankzero():
                monitor_verr.add(i * n_devices, ve.d.copy())
                monitor_vtime.add(i * n_devices)

        # Save model
        if single_or_rankzero():
            if i % (args.model_save_interval // n_devices) == 0:
                iter = i * n_devices
                nn.save_parameters(os.path.join(
                    args.model_save_path, 'params_%06d.h5' % iter))
                if args.use_latest_checkpoint:
                    save_checkpoint(args.model_save_path, iter, solver)

        # Forward/Zerograd
        image, label = tdata.next()
        image_train.d = image
        label_train.d = label
        loss_error_train.forward(clear_no_need_grad=True)
        solver.zero_grad()

        # Backward/AllReduce
        backward_and_all_reduce(
            loss_error_train, comm, with_all_reduce_callback=args.with_all_reduce_callback)

        # Solvers update
        solver.update()

        # Linear Warmup
        if i <= warmup_iter:
            lr = base_lr + warmup_slope * i
            solver.set_learning_rate(lr)

        # Monitoring loss, error and elapsed time
        if single_or_rankzero():
            monitor_loss.add(i * n_devices, loss_train.d.copy())
            monitor_err.add(i * n_devices, error_train.d.copy())
            monitor_time.add(i * n_devices)

    # Save nnp last epoch
    if single_or_rankzero():
        runtime_contents = {
            'networks': [{
                'name': 'Validation',
                'batch_size': args.batch_size,
                'outputs': {'y': pred_valid},
                'names': {'x': image_valid}
            }],
            'executors': [{
                'name': 'Runtime',
                'network': 'Validation',
                'data': ['x'],
                'output':['y']
            }]
        }
        iter = args.epochs * iter_per_epoch
        nn.save_parameters(os.path.join(
            args.model_save_path, 'params_%06d.h5' % iter))
        nnabla.utils.save.save(os.path.join(
            args.model_save_path, f'{args.net}_result.nnp'), runtime_contents)
    if comm:
        comm.barrier()


if __name__ == '__main__':
    """
    Call this script simply for single device training.
    $ python classification.py --context cudnn -b 64

    Call this script with `mpirun` or `mpiexec` for multi device training.
    $ mpirun -n 4 python classification.py --context cudnn -b 64
    """
    train(get_args())
