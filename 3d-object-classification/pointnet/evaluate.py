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

from typing import Dict, Tuple
import argparse
import os

import nnabla as nn
from nnabla.ext_utils import get_extension_context
from nnabla.utils.data_iterator import DataIterator
from nnabla.logger import logger

from model import pointnet_classification
from loss import classification_loss_with_orthogonal_loss
from data.modelnet40_normal_resampled_dataiter import data_iterator_modelnet40_normal_resampled
from running_utils import categorical_accuracy, load_snapshot, set_global_seed


def eval_one_epoch(
    valid_data_iter: DataIterator,
    valid_vars: Dict[str, nn.Variable],
    valid_loss_vars: Dict[str, nn.Variable],
) -> Tuple[str, str]:
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

    # Load snapshot
    load_snapshot(args.snapshot_dir)

    # Data Iterator
    valid_data_iter = data_iterator_modelnet40_normal_resampled(
        args.data_dir, valid_batch_size, False, False, args.num_points, normalize=True, stop_exhausted=True
    )

    # Training-loop
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
    parser.add_argument("--num_classes", type=int, default=40)
    parser.add_argument("--num_points", type=int, default=1024)

    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--context", type=str, default="cudnn")
    parser.add_argument("--snapshot_dir", type=str,
                        default="./pointnet_classification_result/seed_100/best")

    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
