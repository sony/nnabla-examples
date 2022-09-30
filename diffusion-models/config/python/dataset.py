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

from dataclasses import dataclass
from typing import List, Union

from omegaconf import MISSING

# todo: using config name to access other config group is unsafe

# Dataset Definition
@dataclass
class DatasetConfig:
    name: str = MISSING
    data_dir: Union[None, str] = None
    dataset_root_dir: Union[None, str] = None
    on_memory: bool = False
    fix_aspect_ratio: bool = True
    random_crop: bool = False
    shuffle_dataset: bool = True
    train: bool = True

    # for class condition
    num_classes: int = 1

    # for text condition
    max_text_length: int = 0
    text_emb_dims: int = 0

    # from other configs
    channel_last: bool = "${model.channel_last}"
    batch_size: int = "${train.batch_size}"
    image_size: List[int] = "${model.image_size}"

# expose config to enable loading from yaml file
from .utils import register_config
register_config(name="dataset", node=DatasetConfig)