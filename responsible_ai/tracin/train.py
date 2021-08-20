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
import functools
import math
import nnabla as nn
import nnabla.functions as F
import nnabla.solvers as S
import nnabla.utils.save as save
import numpy as np

from nnabla.ext_utils import get_extension_context
from nnabla.utils.data_iterator import data_iterator
from args import get_args
from cifar10_load import data_source_cifar10, Cifar10NumpySource
from model import resnet56_prediction, loss_function, resnet23_prediction

sys.path.append("../../utils/")
from neu.learning_rate_scheduler import create_learning_rate_scheduler
from neu.yaml_wrapper import read_yaml
from neu.save_nnp import save_nnp
from neu.checkpoint_util import save_checkpoint, load_checkpoint


def set_seed(seed=200):
    np.random.seed(seed)
    rng = np.random.seed(seed)
    return rng, seed


def load_data(bs_train, bs_valid):
    x_train = np.load(os.path.join(args.input, "x_train.npy"))
    y_train = np.load(os.path.join(args.input, "y_shuffle_train.npy"))

    x_val = np.load(os.path.join(args.input, "x_val.npy"))
    y_val = np.load(os.path.join(args.input, "y_val.npy"))

    data_source_train = Cifar10NumpySource(x_train, y_train, shuffle=False)
    data_source_val = Cifar10NumpySource(x_val, y_val)

    train_samples, val_samples = len(data_source_train.labels), len(
        data_source_val.labels
    )
    train_loader = data_iterator(
        data_source_train, bs_train, None, False, False)
    val_loader = data_iterator(data_source_val, bs_valid, None, False, False)

    return train_loader, val_loader, train_samples, val_samples


def categorical_error(pred, label):
    """
    Compute categorical error given score vectors and labels as
    numpy.ndarray.
    """
    pred_label = pred.argmax(1)
    return (pred_label != label.flat).mean()


def train():
    bs_train, bs_valid = args.train_batch_size, args.val_batch_size
    extension_module = args.context
    ctx = get_extension_context(
        extension_module, device_id=args.device_id, type_config=args.type_config
    )
    nn.set_default_context(ctx)

    if args.input:
        train_loader, val_loader, n_train_samples, n_val_samples = load_data(
            bs_train, bs_valid
        )

    else:
        train_data_source = data_source_cifar10(
            train=True, shuffle=True, label_shuffle=True
        )
        val_data_source = data_source_cifar10(train=False, shuffle=False)
        n_train_samples = len(train_data_source.labels)
        n_val_samples = len(val_data_source.labels)
        # Data Iterator
        train_loader = data_iterator(
            train_data_source, bs_train, None, False, False)
        val_loader = data_iterator(
            val_data_source, bs_valid, None, False, False)

        if args.shuffle_label:
            if not os.path.exists(args.output):
                os.makedirs(args.output)
            np.save(os.path.join(args.output, "x_train.npy"),
                    train_data_source.images)
            np.save(
                os.path.join(args.output, "y_shuffle_train.npy"),
                train_data_source.labels,
            )
            np.save(os.path.join(args.output, "y_train.npy"),
                    train_data_source.raw_label)
            np.save(os.path.join(args.output, "x_val.npy"),
                    val_data_source.images)
            np.save(os.path.join(args.output, "y_val.npy"),
                    val_data_source.labels)

    if args.model == "resnet23":
        model_prediction = resnet23_prediction
    elif args.model == "resnet56":
        model_prediction = resnet56_prediction
    prediction = functools.partial(
        model_prediction, ncls=10, nmaps=64, act=F.relu, seed=args.seed)

    # Create training graphs
    test = False
    image_train = nn.Variable((bs_train, 3, 32, 32))
    label_train = nn.Variable((bs_train, 1))
    pred_train, _ = prediction(image_train, test)

    loss_train = loss_function(pred_train, label_train)

    # Create validation graph
    test = True
    image_valid = nn.Variable((bs_valid, 3, 32, 32))
    label_valid = nn.Variable((bs_valid, 1))
    pred_valid, _ = prediction(image_valid, test)
    loss_val = loss_function(pred_valid, label_valid)

    for param in nn.get_parameters().values():
        param.grad.zero()

    cfg = read_yaml("./learning_rate.yaml")
    print(cfg)
    lr_sched = create_learning_rate_scheduler(cfg.learning_rate_config)
    solver = S.Momentum(momentum=0.9, lr=lr_sched.get_lr())
    solver.set_parameters(nn.get_parameters())
    start_point = 0

    if args.checkpoint is not None:
        # load weights and solver state info from specified checkpoint file.
        start_point = load_checkpoint(args.checkpoint, solver)

    # Create monitor
    from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed

    monitor = Monitor(args.monitor_path)
    monitor_loss = MonitorSeries("Training loss", monitor, interval=1)
    monitor_err = MonitorSeries("Training error", monitor, interval=1)
    monitor_time = MonitorTimeElapsed("Training time", monitor, interval=1)
    monitor_verr = MonitorSeries("Test error", monitor, interval=1)
    monitor_vloss = MonitorSeries("Test loss", monitor, interval=1)

    # save_nnp
    contents = save_nnp({"x": image_valid}, {"y": pred_valid}, bs_valid)
    save.save(
        os.path.join(args.model_save_path,
                     (args.model+"_epoch0_result.nnp")), contents
    )

    train_iter = math.ceil(n_train_samples / bs_train)
    val_iter = math.ceil(n_val_samples / bs_valid)

    # Training-loop
    for i in range(start_point, args.train_epochs):
        lr_sched.set_epoch(i)
        solver.set_learning_rate(lr_sched.get_lr())
        print("Learning Rate: ", lr_sched.get_lr())
        # Validation
        ve = 0.0
        vloss = 0.0
        print("## Validation")
        for j in range(val_iter):
            image, label = val_loader.next()
            image_valid.d = image
            label_valid.d = label
            loss_val.forward()
            vloss += loss_val.data.data.copy() * bs_valid
            ve += categorical_error(pred_valid.d, label)
        ve /= args.val_iter
        vloss /= n_val_samples

        monitor_verr.add(i, ve)
        monitor_vloss.add(i, vloss)

        if int(i % args.model_save_interval) == 0:
            # save checkpoint file
            save_checkpoint(args.model_save_path, i, solver)

        # Forward/Zerograd/Backward
        print("## Training")
        e = 0.0
        loss = 0.0
        for k in range(train_iter):

            image, label = train_loader.next()
            image_train.d = image
            label_train.d = label
            loss_train.forward()
            solver.zero_grad()
            loss_train.backward()
            solver.update()
            e += categorical_error(pred_train.d, label_train.d)
            loss += loss_train.data.data.copy() * bs_train
        e /= train_iter
        loss /= n_train_samples

        e = categorical_error(pred_train.d, label_train.d)
        monitor_loss.add(i, loss)
        monitor_err.add(i, e)
        monitor_time.add(i)

    nn.save_parameters(
        os.path.join(args.model_save_path, "params_%06d.h5" %
                     (args.train_epochs))
    )

    # save_nnp_lastepoch
    contents = save_nnp({"x": image_valid}, {"y": pred_valid}, bs_valid)
    save.save(os.path.join(args.model_save_path,
              (args.model+"_result.nnp")), contents)


if __name__ == "__main__":
    """
    $ python train.py --context "cudnn" -b 64

    """
    args = get_args()
    train()
