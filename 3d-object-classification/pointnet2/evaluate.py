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

from typing import Dict, Tuple
import argparse
import os
import numpy as np

import nnabla as nn
from nnabla.ext_utils import get_extension_context
from nnabla.utils.data_iterator import DataIterator
from nnabla.logger import logger

from pointnet2 import pointnet2_classification_msg, pointnet2_classification_ssg
from loss import classification_loss
from running_utils import categorical_accuracy

# Install neu (nnabla examples utils) to import these functions.
# See [NEU](https://github.com/nnabla/nnabla-examples/tree/master/utils).
from neu.datasets.modelnet40_normal_resampled import data_iterator_modelnet40_normal_resampled
from neu.checkpoint_util import load_checkpoint


def eval_one_epoch(
    valid_data_iter: DataIterator,
    valid_vars: Dict[str, nn.Variable],
    valid_loss_vars: Dict[str, nn.Variable],
) -> Tuple[np.ndarray, np.ndarray]:
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

    average_accuracy = total_accuracy / float(total_steps)
    average_loss = total_loss / float(total_steps)

    return average_accuracy, average_loss


def evaluate(args):
    # Set context
    extension_module = args.context
    ctx = get_extension_context(extension_module, device_id=args.device_id)
    nn.set_default_context(ctx)

    # Feature dim, with normal vector or not
    feature_dim = 6 if args.with_normal else 3

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

    # Load snapshot
    load_checkpoint(args.checkpoint_json_path, {})

    # Data Iterator
    valid_data_iter = data_iterator_modelnet40_normal_resampled(
        args.data_dir,
        valid_batch_size,
        False,
        False,
        args.num_points,
        normalize=True,
        with_normal=args.with_normal,
    )
    logger.info(f"Validation dataset size: {valid_data_iter.size}")

    # Evaluation
    logger.info(f"Evaluation starting ...")
    accuracy, loss = eval_one_epoch(
        valid_data_iter, valid_vars, valid_loss_vars)
    logger.info("accuracy: {}".format(accuracy))
    logger.info("loss: {}".format(loss))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default=os.path.join(os.path.dirname(__file__), "data", "modelnet40_normal_resampled")
    )
    parser.add_argument("--model_type", type=str,
                        default="ssg", choices=["msg", "ssg"])
    parser.add_argument("--num_classes", type=int, default=40)
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--with_normal", action="store_true")

    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--context", type=str, default="cudnn")
    parser.add_argument(
        "--checkpoint_json_path",
        type=str,
        default="./pointnet2_classification_result/seed_100/checkpoint_best/checkpoint_best.json",
    )

    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
