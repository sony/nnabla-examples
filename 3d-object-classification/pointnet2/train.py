# Copyright 2022 Sony Group Corporation.
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

from typing import Dict
import argparse
import os
import numpy as np

import nnabla as nn
from nnabla.ext_utils import get_extension_context
from nnabla.utils.data_iterator import DataIterator
import nnabla.solvers as S
from nnabla.logger import logger
from nnabla.monitor import Monitor, MonitorSeries

from pointnet2 import pointnet2_classification_msg, pointnet2_classification_ssg
from loss import classification_loss
from running_utils import categorical_accuracy, set_global_seed

# Install neu (nnabla examples utils) to import these functions.
# See [NEU](https://github.com/nnabla/nnabla-examples/tree/master/utils).
from neu.datasets.modelnet40_normal_resampled import data_iterator_modelnet40_normal_resampled
from neu.checkpoint_util import save_checkpoint
from neu.learning_rate_scheduler import create_learning_rate_scheduler
from neu.misc import AttrDict


def train_one_epoch(
    train_data_iter: DataIterator,
    train_vars: Dict[str, nn.Variable],
    train_loss_vars: Dict[str, nn.Variable],
    solver: S.Solver,
    weight_decay: float,
    learning_rate: float,
    train_monitors: Dict[str, Monitor],
    global_steps: int,
) -> int:
    total_steps = global_steps
    num_iterations = train_data_iter.size // train_data_iter.batch_size

    for _ in range(num_iterations):
        point_cloud, label = train_data_iter.next()

        train_vars["point_cloud"].d = point_cloud
        train_vars["label"].d = label
        train_loss_vars["loss"].forward(clear_no_need_grad=True)
        solver.zero_grad()
        train_loss_vars["loss"].backward(clear_buffer=True)

        solver.set_learning_rate(learning_rate)
        solver.weight_decay(weight_decay)
        solver.update()

        accuracy = categorical_accuracy(
            train_loss_vars["pred"].d, train_vars["label"].d)
        train_monitors["loss"].add(
            total_steps, train_loss_vars["loss"].d.copy())
        train_monitors["accuracy"].add(total_steps, accuracy)
        total_steps += 1

    return total_steps


def eval_one_epoch(
    valid_data_iter: DataIterator,
    valid_vars: Dict[str, nn.Variable],
    valid_loss_vars: Dict[str, nn.Variable],
    valid_monitors: Dict[str, Monitor],
    global_steps: int,
) -> float:
    total_steps = 0
    total_accuracy = 0.0
    total_loss = 0.0
    num_iterations = valid_data_iter.size // valid_data_iter.batch_size

    for _ in range(num_iterations):
        point_cloud, label = valid_data_iter.next()

        valid_vars["point_cloud"].d = point_cloud
        valid_vars["label"].d = label
        valid_loss_vars["loss"].forward(clear_buffer=True)

        pred_logits = valid_loss_vars["pred"].d.copy()

        accuracy = categorical_accuracy(pred_logits, valid_vars["label"].d)
        total_steps += 1
        total_accuracy += accuracy
        total_loss += float(valid_loss_vars["loss"].d)

    valid_monitors["accuracy"].add(
        global_steps, total_accuracy / float(total_steps))
    valid_monitors["loss"].add(global_steps, total_loss / float(total_steps))

    return total_accuracy / float(total_steps)


def train(args):
    # Create out dir
    outdir = os.path.join(args.result_dir, f"seed_{args.seed}")
    os.makedirs(outdir, exist_ok=True)

    # Set context
    extension_module = args.context
    ctx = get_extension_context(extension_module, device_id=args.device_id)
    nn.set_default_context(ctx)

    # Set seed
    set_global_seed(args.seed)

    # Feature dim, with normal vector or not
    feature_dim = 6 if args.with_normal else 3

    # Create training graphs
    point_cloud_train = nn.Variable(
        (args.batch_size, args.num_points, feature_dim))
    label_train = nn.Variable((args.batch_size, 1))

    if args.model_type == "ssg":
        pred_train = pointnet2_classification_ssg(
            point_cloud_train, train=True, num_classes=args.num_classes)
    elif args.model_type == "msg":
        pred_train = pointnet2_classification_msg(
            point_cloud_train, train=True, num_classes=args.num_classes)
    else:
        raise ValueError

    pred_train.persistent = True
    loss_train = classification_loss(pred_train, label_train)
    train_vars = {"point_cloud": point_cloud_train, "label": label_train}
    train_loss_vars = {"loss": loss_train, "pred": pred_train}

    # Create validation graph
    valid_batch_size = 4  # Setting 4 is for using all data of valid dataset
    point_cloud_valid = nn.Variable(
        (valid_batch_size, args.num_points, feature_dim))
    label_valid = nn.Variable((valid_batch_size, 1))

    if args.model_type == "ssg":
        pred_valid = pointnet2_classification_ssg(
            point_cloud_valid, train=False, num_classes=args.num_classes)
    elif args.model_type == "msg":
        pred_valid = pointnet2_classification_msg(
            point_cloud_valid, train=False, num_classes=args.num_classes)
    else:
        raise ValueError

    pred_valid.persistent = True
    loss_valid = classification_loss(pred_valid, label_valid)
    valid_vars = {"point_cloud": point_cloud_valid, "label": label_valid}
    valid_loss_vars = {"loss": loss_valid, "pred": pred_valid}

    # Solvers
    solver = S.Adam(args.learning_rate)
    solver.set_parameters(nn.get_parameters())

    # Create monitor
    monitor_path = os.path.join(outdir, "monitors")
    monitor = Monitor(monitor_path)
    train_monitors = {}
    train_monitors["loss"] = MonitorSeries(
        "training loss", monitor, interval=10)
    train_monitors["accuracy"] = MonitorSeries(
        "training accuracy", monitor, interval=10)

    valid_monitors = {}
    valid_monitors["loss"] = MonitorSeries("valid loss", monitor, interval=1)
    valid_monitors["accuracy"] = MonitorSeries(
        "valid accuracy", monitor, interval=1)

    # Data Iterator
    train_data_iter = data_iterator_modelnet40_normal_resampled(
        args.data_dir,
        args.batch_size,
        True,
        True,
        args.num_points,
        normalize=True,
        with_normal=args.with_normal,
        rng=args.seed,
    )
    logger.info(f"Training dataset size: {train_data_iter.size}")
    valid_data_iter = data_iterator_modelnet40_normal_resampled(
        args.data_dir,
        valid_batch_size,
        False,
        False,
        args.num_points,
        normalize=True,
        with_normal=args.with_normal,
        rng=args.seed,
    )
    logger.info(f"Validation dataset size: {valid_data_iter.size}")

    # Training-loop
    global_steps = 0
    decayed_learning_rate = args.learning_rate
    best_accuracy = 0.0
    learning_rate_config = {
        "scheduler_type": "EpochStepLearningRateScheduler",
        "base_lr": args.learning_rate,
        "decay_at": np.arange(20, args.max_epoch, 20),
        "decay_rate": 0.7,
        "warmup_epochs": 0,
    }
    learning_rate_scheduler = create_learning_rate_scheduler(
        AttrDict(learning_rate_config))

    for i in range(1, args.max_epoch + 1):
        logger.info(f"Training {i} th epoch...")
        decayed_learning_rate = learning_rate_scheduler.get_lr_and_update()

        global_steps = train_one_epoch(
            train_data_iter,
            train_vars,
            train_loss_vars,
            solver,
            args.weight_decay,
            decayed_learning_rate,
            train_monitors,
            global_steps,
        )

        learning_rate_scheduler.set_epoch(i + 1)

        if i % args.eval_interval == 0:
            logger.info(f"Evaluation at {i} th epoch ...")
            accuracy = eval_one_epoch(
                valid_data_iter, valid_vars, valid_loss_vars, valid_monitors, global_steps)
            checkpoint_outdir = os.path.join(outdir, f"checkpoint_{i}")
            os.makedirs(checkpoint_outdir, exist_ok=True)
            save_checkpoint(checkpoint_outdir, i, solver)

            if accuracy > best_accuracy:
                logger.info("Update best and save current parameters")
                best_accuracy = accuracy
                best_outdir = os.path.join(outdir, "checkpoint_best")
                os.makedirs(best_outdir, exist_ok=True)
                save_checkpoint(best_outdir, "best", solver)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default=os.path.join(os.path.dirname(__file__), "data", "modelnet40_normal_resampled")
    )
    parser.add_argument("--model_type", type=str,
                        default="msg", choices=["msg", "ssg"])
    parser.add_argument("--num_classes", type=int, default=40)
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--with_normal", action="store_true")

    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--max_epoch", type=int, default=250)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--context", type=str, default="cudnn")
    parser.add_argument("--result_dir", type=str,
                        default="pointnet2_classification_result")
    parser.add_argument("--seed", type=int, default=100)

    parser.add_argument("--eval_interval", type=int, default=2)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
