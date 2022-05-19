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

from typing import Tuple
import numpy as np
import nnabla as nn
import random as py_random

from nnabla.logger import logger

# Install neu (nnabla examples utils) to import these functions.
# See [NEU](https://github.com/nnabla/nnabla-examples/tree/master/utils).
from neu.datasets.shapenet_partanno_segmentation import (
    SEGMENTATION_ID_TO_CLASS_NAME_DICT,
    CLASS_NAME_TO_SEGMENTATION_ID_DICT,
)


def categorical_accuracy(pred: np.ndarray, label: np.ndarray) -> np.ndarray:
    pred_label = np.argmax(pred.reshape(-1, pred.shape[-1]), axis=1)
    return (pred_label == label.flatten()).mean()


def to_one_hot(class_label: np.ndarray, num_classes: int) -> np.ndarray:
    return np.eye(num_classes)[class_label.flatten()]


def take_argmax_in_each_class(pred_logits: np.ndarray, segmentation_label: np.ndarray) -> np.ndarray:
    batch_size, num_points, _ = pred_logits.shape
    pred_each_class_seg_id = np.zeros((batch_size, num_points))
    for i in range(batch_size):
        class_name = SEGMENTATION_ID_TO_CLASS_NAME_DICT[segmentation_label[i, 0]]
        each_pred_logits = pred_logits[i, :, :]  # shape(num_points, num_parts)
        pred_each_class_seg_id[i, :] = (
            np.argmax(
                each_pred_logits[:, CLASS_NAME_TO_SEGMENTATION_ID_DICT[class_name]], 1)
            + CLASS_NAME_TO_SEGMENTATION_ID_DICT[class_name][0]
        )
    return pred_each_class_seg_id


def compute_shape_iou(pred_seg_id: np.ndarray, seg_id: np.ndarray) -> Tuple[str, np.ndarray]:
    class_name = SEGMENTATION_ID_TO_CLASS_NAME_DICT[seg_id[0]]
    part_ious = np.zeros(len(CLASS_NAME_TO_SEGMENTATION_ID_DICT[class_name]))

    for class_seg_id in CLASS_NAME_TO_SEGMENTATION_ID_DICT[class_name]:
        seg_id_index = class_seg_id - \
            CLASS_NAME_TO_SEGMENTATION_ID_DICT[class_name][0]
        if (np.sum(pred_seg_id == class_seg_id) == 0) and (
            np.sum(seg_id == class_seg_id) == 0
        ):  # NOTE: should be 1, if part is not present, no prediction as well
            part_ious[seg_id_index] = 1.0
        else:  # Compute IoU of a part
            part_ious[seg_id_index] = np.sum((pred_seg_id == class_seg_id) & (seg_id == class_seg_id)) / float(
                np.sum((pred_seg_id == class_seg_id)
                       | (seg_id == class_seg_id))
            )
    return class_name, np.mean(part_ious)


def set_global_seed(seed: int) -> None:
    np.random.seed(seed=seed)
    py_random.seed(seed)
    nn.seed(seed)
    logger.info("Set seed to {}".format(seed))
