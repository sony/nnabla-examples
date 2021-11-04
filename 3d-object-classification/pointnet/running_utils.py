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
import random as py_random
import numpy as np

import nnabla as nn
from nnabla.logger import logger


def set_global_seed(seed: int) -> None:
    np.random.seed(seed=seed)
    py_random.seed(seed)
    nn.seed(seed)
    logger.info("Set seed to {}".format(seed))


def save_snapshot(save_dir: str) -> None:
    logger.info("Save network parameters")
    os.makedirs(save_dir, exist_ok=True)
    model_file_path = os.path.join(save_dir, "pointnet_classification.h5")
    nn.save_parameters(path=model_file_path)


def load_snapshot(load_dir: str, file_name: str = "pointnet_classification.h5") -> None:
    logger.info("Load network parameters")
    model_file_path = os.path.join(load_dir, file_name)
    nn.load_parameters(path=model_file_path)


def categorical_accuracy(pred: np.ndarray, label: np.ndarray) -> np.ndarray:
    pred_label = np.argmax(pred, axis=1)
    return (pred_label == label.flatten()).mean()


def get_decayed_learning_rate(
    num_epoch: int, learning_rate: float, decay_step: int = 20, decay_rate: float = 0.7
) -> float:
    if num_epoch % decay_step == 0:
        learning_rate *= decay_rate
    return learning_rate
