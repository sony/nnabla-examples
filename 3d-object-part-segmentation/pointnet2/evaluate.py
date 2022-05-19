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
from typing import Dict, Tuple
import argparse
import os
import numpy as np

import nnabla as nn
from nnabla.ext_utils import get_extension_context
from nnabla.utils.data_iterator import DataIterator
from nnabla.logger import logger

from pointnet2 import pointnet2_part_segmentation_msg, pointnet2_part_segmentation_ssg
from loss import classification_loss
from running_utils import categorical_accuracy, compute_shape_iou, to_one_hot, take_argmax_in_each_class

# Install neu (nnabla examples utils) to import these functions.
# See [NEU](https://github.com/nnabla/nnabla-examples/tree/master/utils).
from neu.datasets.shapenet_partanno_segmentation import data_iterator_shapenet_partanno_segmentation
from neu.checkpoint_util import load_checkpoint


def eval_one_epoch(
    valid_data_iter: DataIterator,
    valid_vars: Dict[str, nn.Variable],
    valid_loss_vars: Dict[str, nn.Variable],
    num_classes: int,
    num_parts: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray], np.ndarray, np.ndarray]:
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

    average_accuracy = total_accuracy / float(total_steps)
    average_loss = total_loss / float(total_steps)
    average_part_accuracy = np.mean(total_part_accuracy / total_part_count)

    # Compute mean of each class shape IoU and mean of all IoU
    all_shape_ious = []
    for class_name in shape_ious.keys():
        for shape_iou in shape_ious[class_name]:
            all_shape_ious.append(shape_iou)
        shape_ious[class_name] = np.mean(shape_ious[class_name])

    return (
        average_loss,
        average_accuracy,
        average_part_accuracy,
        shape_ious,
        np.mean(list(shape_ious.values())),
        np.mean(all_shape_ious),
    )


def evaluate(args):
    # Set context
    extension_module = args.context
    ctx = get_extension_context(extension_module, device_id=args.device_id)
    nn.set_default_context(ctx)

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

    # Load snapshot
    load_checkpoint(args.checkpoint_json_path, {})

    # Data Iterator
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
    )
    logger.info(f"Validation dataset size: {valid_data_iter.size}")

    # Evaluation
    logger.info(f"Evaluation starting ...")
    loss, accuracy, part_accuracy, shape_ious, mean_iou_class, mean_iou_shape = eval_one_epoch(
        valid_data_iter,
        valid_vars,
        valid_loss_vars,
        args.num_classes,
        args.num_parts,
    )
    logger.info("loss: {}".format(loss))
    logger.info("accuracy: {}".format(accuracy))
    logger.info("part accuracy: {}".format(part_accuracy))
    logger.info("Mean IoU (shape average): {}".format(mean_iou_shape))
    logger.info("Mean IoU (class average): {}".format(mean_iou_class))
    logger.info("IoU of each class: {}".format(shape_ious))


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

    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--context", type=str, default="cudnn")
    parser.add_argument(
        "--checkpoint_json_path",
        type=str,
        default="./pointnet2_part-segmentation_result/seed_100/checkpoint_best/checkpoint_best.json",
    )

    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
