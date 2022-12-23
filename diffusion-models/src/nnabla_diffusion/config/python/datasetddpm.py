# Copyright 2023 Sony Group Corporation.
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

from .utils import register_config
from dataclasses import dataclass
from typing import List, Union

from omegaconf import MISSING


@dataclass
class DatasetDDPMConfig:
    h5: str = MISSING
    config: str = MISSING

    seed: int = 0
    ema: bool = True
    category: str = "ffhq_34"
    training_path: str = MISSING
    validation_path: str = MISSING
    testing_path: str = MISSING

    dim: List[int] = MISSING
    blocks: List[int] = MISSING
    steps: List[int] = MISSING

    upsampling_mode: str = "linear"  # what is the difference from "bilinear"
    model_num: int = 10

    training_number: int = 30
    testing_number: int = 30
    max_training: int = 30
    batch_size: int = 64

    image_size: int = 256
    ignore_label: int = 255
    number_class: int = 34
    use_bn: bool = True

    share_noise: bool = True
    shuffle_dataset: bool = True
    float = MISSING

    output_dir: str = "./segmentation_results"


# expose config to enable loading from yaml file

register_config(name="datasetddpm", node=DatasetDDPMConfig)
