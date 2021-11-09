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

from typing import Dict
import argparse
import os

import nnabla as nn
from nnabla.ext_utils import get_extension_context
from nnabla.utils.data_iterator import DataIterator
import nnabla.solvers as S
from nnabla.logger import logger
from nnabla.monitor import Monitor, MonitorSeries

from model import pointnet_classification
from loss import classification_loss_with_orthogonal_loss
from data.modelnet40_normal_resampled_dataiter import data_iterator_modelnet40_normal_resampled
from running_utils import categorical_accuracy, save_snapshot, set_global_seed, get_decayed_learning_rate


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
    train_data_iter._reset()

    for batch_data in train_data_iter:
        point_cloud, label = batch_data

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
        train_monitors["loss/mat_loss"].add(total_steps,
                                            train_loss_vars["mat_loss"].d.copy())
        train_monitors["loss/classify_loss"].add(
            total_steps, train_loss_vars["classify_loss"].d.copy())
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
    valid_data_iter._reset()

    for batch_data in valid_data_iter:
        point_cloud, label = batch_data

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

    # Create training graphs
    point_cloud_train = nn.Variable((args.batch_size, args.num_points, 3))
    label_train = nn.Variable((args.batch_size, 1))
    pred_train, internal_train_variables = pointnet_classification(
        point_cloud_train, train=True, num_classes=args.num_classes
    )
    pred_train.persistent = True
    loss_train, internal_losses_train = classification_loss_with_orthogonal_loss(
        pred_train,
        label_train,
        internal_train_variables["pointnet_feature_internal_variables"]["feature_transformation_mat"],
    )
    internal_losses_train["mat_loss"].persistent = True
    internal_losses_train["classify_loss"].persistent = True
    train_vars = {"point_cloud": point_cloud_train, "label": label_train}
    train_loss_vars = {"loss": loss_train,
                       "pred": pred_train, **internal_losses_train}

    # Create validation graph
    valid_batch_size = 4  # Setting 4 is for using all data of valid dataset
    point_cloud_valid = nn.Variable((valid_batch_size, args.num_points, 3))
    label_valid = nn.Variable((valid_batch_size, 1))
    pred_valid, internal_valid_variables = pointnet_classification(
        point_cloud_valid, train=False, num_classes=args.num_classes
    )
    pred_valid.persistent = True
    loss_valid, internal_losses_valid = classification_loss_with_orthogonal_loss(
        pred_valid,
        label_valid,
        internal_valid_variables["pointnet_feature_internal_variables"]["feature_transformation_mat"],
    )
    valid_vars = {"point_cloud": point_cloud_valid, "label": label_valid}
    valid_loss_vars = {"loss": loss_valid,
                       "pred": pred_valid, **internal_losses_valid}

    # Solvers
    solver = S.Adam(args.learning_rate)
    solver.set_parameters(nn.get_parameters())

    # Create monitor
    monitor_path = os.path.join(outdir, "monitors")
    monitor = Monitor(monitor_path)
    train_monitors = {}
    train_monitors["loss"] = MonitorSeries(
        "training loss", monitor, interval=10)
    train_monitors["loss/mat_loss"] = MonitorSeries(
        "training matrix loss", monitor, interval=10)
    train_monitors["loss/classify_loss"] = MonitorSeries(
        "training classification loss", monitor, interval=10)
    train_monitors["accuracy"] = MonitorSeries(
        "training accuracy", monitor, interval=10)

    valid_monitors = {}
    valid_monitors["loss"] = MonitorSeries("valid loss", monitor, interval=1)
    valid_monitors["accuracy"] = MonitorSeries(
        "valid accuracy", monitor, interval=1)

    # Data Iterator
    train_data_iter = data_iterator_modelnet40_normal_resampled(
        args.data_dir, args.batch_size, True, True, args.num_points, normalize=True, stop_exhausted=True
    )
    valid_data_iter = data_iterator_modelnet40_normal_resampled(
        args.data_dir, valid_batch_size, False, False, args.num_points, normalize=True, stop_exhausted=True
    )

    # Training-loop
    global_steps = 0
    decayed_learning_rate = args.learning_rate
    best_accuracy = 0.0

    for i in range(1, args.max_epoch + 1):
        logger.info(f"Training {i} th epoch...")
        decayed_learning_rate = get_decayed_learning_rate(
            i, decayed_learning_rate)

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

        if i % args.eval_interval == 0:
            logger.info(f"Evaluation at {i} th epoch ...")
            accuracy = eval_one_epoch(
                valid_data_iter, valid_vars, valid_loss_vars, valid_monitors, global_steps)
            save_dir = os.path.join(outdir, "epoch_{}".format(i))
            save_snapshot(save_dir)

            if accuracy > best_accuracy:
                logger.info("Update best and save current parameters")
                best_accuracy = accuracy
                save_dir = os.path.join(outdir, "best")
                save_snapshot(save_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default=os.path.join(os.path.dirname(__file__), "data", "modelnet40_normal_resampled")
    )
    parser.add_argument("--num_classes", type=int, default=40)
    parser.add_argument("--num_points", type=int, default=1024)

    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--max_epoch", type=int, default=250)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--context", type=str, default="cudnn")
    parser.add_argument("--result_dir", type=str,
                        default="pointnet_classification_result")
    parser.add_argument("--seed", type=int, default=100)

    parser.add_argument("--eval_interval", type=int, default=2)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
