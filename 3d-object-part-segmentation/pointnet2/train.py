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

from collections import defaultdict
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

from pointnet2 import pointnet2_part_segmentation_msg, pointnet2_part_segmentation_ssg
from loss import classification_loss
from running_utils import (
    categorical_accuracy,
    set_global_seed,
    compute_shape_iou,
    to_one_hot,
    take_argmax_in_each_class,
)

# Install neu (nnabla examples utils) to import these functions.
# See [NEU](https://github.com/nnabla/nnabla-examples/tree/master/utils).
from neu.datasets.shapenet_partanno_segmentation import data_iterator_shapenet_partanno_segmentation
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
    num_classes: int,
) -> int:
    total_steps = global_steps
    num_iterations = train_data_iter.size // train_data_iter.batch_size

    for _ in range(num_iterations):
        point_cloud, class_label, segmentation_label = train_data_iter.next()

        train_vars["point_cloud"].d = point_cloud
        train_vars["one_hot_class_label"].d = to_one_hot(
            class_label, num_classes)
        train_vars["segmentation_label"].d = segmentation_label
        train_loss_vars["loss"].forward(clear_no_need_grad=True)
        solver.zero_grad()
        train_loss_vars["loss"].backward(clear_buffer=True)

        solver.set_learning_rate(learning_rate)
        solver.weight_decay(weight_decay)
        solver.update()

        accuracy = categorical_accuracy(
            train_loss_vars["pred"].d, train_vars["segmentation_label"].d)
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
    num_classes: int,
    num_parts: int,
) -> np.ndarray:
    total_steps = 0
    total_accuracy = 0.0
    total_part_accuracy = np.zeros(num_parts)
    total_part_count = np.zeros(num_parts)
    total_loss = 0.0
    num_iterations = valid_data_iter.size // valid_data_iter.batch_size
    shape_ious = defaultdict(list)

    for _ in range(num_iterations):
        point_cloud, class_label, segmentation_label = valid_data_iter.next()

        valid_vars["point_cloud"].d = point_cloud
        valid_vars["one_hot_class_label"].d = to_one_hot(
            class_label, num_classes)
        valid_vars["segmentation_label"].d = segmentation_label
        valid_loss_vars["loss"].forward(clear_no_need_grad=True)

        pred_logits = valid_loss_vars["pred"].d.copy()

        accuracy = categorical_accuracy(pred_logits, segmentation_label)

        # Compute accuracy of each class
        pred_each_class_seg_id = take_argmax_in_each_class(
            pred_logits, segmentation_label)

        # Compute each part accuracy
        for i in range(num_parts):
            total_part_count[i] += np.sum(segmentation_label == i)
            total_part_accuracy[i] += np.sum(
                (pred_each_class_seg_id == i) & (segmentation_label == i))

        # Compute shape IoU
        for i in range(valid_data_iter.batch_size):
            class_name, shape_iou = compute_shape_iou(
                pred_each_class_seg_id[i, :], segmentation_label[i, :])
            shape_ious[class_name].append(shape_iou)

        total_steps += 1
        total_accuracy += accuracy
        total_loss += float(valid_loss_vars["loss"].d)

    # Compute mean of each class shape IoU and mean of all IoU
    all_shape_ious = []
    for class_name in shape_ious.keys():
        for shape_iou in shape_ious[class_name]:
            all_shape_ious.append(shape_iou)
        shape_ious[class_name] = np.mean(shape_ious[class_name])

    logger.info(f"IoU of each class {shape_ious}")
    valid_monitors["mean_iou_class"].add(
        global_steps, np.mean(list(shape_ious.values())))
    valid_monitors["mean_iou_shape"].add(global_steps, np.mean(all_shape_ious))
    valid_monitors["accuracy"].add(
        global_steps, total_accuracy / float(total_steps))
    valid_monitors["part_accuracy"].add(
        global_steps, np.mean(total_part_accuracy / total_part_count))
    valid_monitors["loss"].add(global_steps, total_loss / float(total_steps))
    return np.mean(all_shape_ious)


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
    # point (3 dim) and normal vector (3 dim)
    point_cloud_train = nn.Variable((args.batch_size, args.num_points, 6))
    one_hot_class_label_train = nn.Variable(
        (args.batch_size, args.num_classes))
    segmentation_train = nn.Variable((args.batch_size, args.num_points))

    if args.model_type == "ssg":
        # pred_train.shape (batch_size, num_points, num_part)
        pred_train = pointnet2_part_segmentation_ssg(
            point_cloud_train,
            one_hot_class_label_train,
            train=True,
            num_parts=args.num_parts,
            num_classes=args.num_classes,
        )
    elif args.model_type == "msg":
        # pred_train.shape (batch_size, num_points, num_part)
        pred_train = pointnet2_part_segmentation_msg(
            point_cloud_train,
            one_hot_class_label_train,
            train=True,
            num_parts=args.num_parts,
            num_classes=args.num_classes,
        )
    else:
        raise ValueError

    pred_train.persistent = True
    loss_train = classification_loss(pred_train, segmentation_train)
    train_vars = {
        "point_cloud": point_cloud_train,
        "one_hot_class_label": one_hot_class_label_train,
        "segmentation_label": segmentation_train,
    }
    train_loss_vars = {"loss": loss_train, "pred": pred_train}

    # Create validation graph
    valid_batch_size = 6  # Setting 6 is for using all data of valid dataset
    # point (3 dim) and normal vector (3 dim)
    point_cloud_valid = nn.Variable((valid_batch_size, args.num_points, 6))
    one_hot_class_label_valid = nn.Variable(
        (valid_batch_size, args.num_classes))
    segmentation_valid = nn.Variable((valid_batch_size, args.num_points))

    if args.model_type == "ssg":
        # pred_valid.shape (batch_size, num_points, num_part)
        pred_valid = pointnet2_part_segmentation_ssg(
            point_cloud_valid,
            one_hot_class_label_valid,
            train=False,
            num_parts=args.num_parts,
            num_classes=args.num_classes,
        )
    elif args.model_type == "msg":
        # pred_valid.shape (batch_size, num_points, num_part)
        pred_valid = pointnet2_part_segmentation_msg(
            point_cloud_valid,
            one_hot_class_label_valid,
            train=False,
            num_parts=args.num_parts,
            num_classes=args.num_classes,
        )
    else:
        raise ValueError

    pred_valid.persistent = True
    loss_valid = classification_loss(pred_valid, segmentation_valid)
    valid_vars = {
        "point_cloud": point_cloud_valid,
        "one_hot_class_label": one_hot_class_label_valid,
        "segmentation_label": segmentation_valid,
    }
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
    valid_monitors["mean_iou_class"] = MonitorSeries(
        "mean iou class", monitor, interval=1)
    valid_monitors["mean_iou_shape"] = MonitorSeries(
        "mean iou shape", monitor, interval=1)
    valid_monitors["part_accuracy"] = MonitorSeries(
        "part accuracy", monitor, interval=1)

    # Data Iterator
    train_data_iter = data_iterator_shapenet_partanno_segmentation(
        args.data_dir,
        args.batch_size,
        True,
        True,
        args.num_points,
        normalize=True,
        shift=True,
        scale=True,
        with_normal=True,
        rng=args.seed,
    )
    logger.info(f"Training dataset size: {train_data_iter.size}")
    valid_data_iter = data_iterator_shapenet_partanno_segmentation(
        args.data_dir,
        valid_batch_size,
        False,
        False,
        args.num_points,
        shift=False,
        scale=False,
        normalize=True,
        with_normal=True,
        rng=args.seed,
    )
    logger.info(f"Validation dataset size: {valid_data_iter.size}")

    # Training-loop
    global_steps = 0
    decayed_learning_rate = args.learning_rate
    best_mean_ios = 0.0
    learning_rate_config = {
        "scheduler_type": "EpochStepLearningRateScheduler",
        "base_lr": args.learning_rate,
        "decay_at": np.arange(20, args.max_epoch, 20),
        "decay_rate": 0.5,
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
            args.num_classes,
        )

        learning_rate_scheduler.set_epoch(i + 1)

        if i % args.eval_interval == 0:
            logger.info(f"Evaluation at {i} th epoch ...")
            mean_iou_shape = eval_one_epoch(
                valid_data_iter,
                valid_vars,
                valid_loss_vars,
                valid_monitors,
                global_steps,
                args.num_classes,
                args.num_parts,
            )
            checkpoint_outdir = os.path.join(outdir, f"checkpoint_{i}")
            os.makedirs(checkpoint_outdir, exist_ok=True)
            save_checkpoint(checkpoint_outdir, i, solver)

            if mean_iou_shape > best_mean_ios:
                logger.info("Update best and save current parameters")
                best_mean_ios = mean_iou_shape
                best_outdir = os.path.join(outdir, "checkpoint_best")
                os.makedirs(best_outdir, exist_ok=True)
                save_checkpoint(best_outdir, "best", solver)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(
            os.path.dirname(
                __file__), "data", "shapenetcore_partanno_segmentation_benchmark_v0_normal"
        ),
    )
    parser.add_argument("--model_type", type=str,
                        default="ssg", choices=["msg", "ssg"])
    parser.add_argument("--num_parts", type=int, default=50)
    parser.add_argument("--num_classes", type=int, default=16)
    parser.add_argument("--num_points", type=int, default=2048)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_epoch", type=int, default=250)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--context", type=str, default="cudnn")
    parser.add_argument("--result_dir", type=str,
                        default="pointnet2_part-segmentation_result")
    parser.add_argument("--seed", type=int, default=100)

    parser.add_argument("--eval_interval", type=int, default=1)

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
