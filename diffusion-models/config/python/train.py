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
from typing import Union

from omegaconf import MISSING

@dataclass
class TrainConfig:
    batch_size: int = MISSING
    accum: int = MISSING
    n_iters: int = 2500000

    # dump
    progress: bool = False
    output_dir: str = "./logdir"
    save_interval: int = 10000
    show_interval: int = 10
    gen_interval: int = 20000
    dump_grad_norm: bool = False

    # checkpointing
    resume: bool = True

    # augmentation
    # If True, Gaussian conditioning augmentation proposed in "Cascaded Diffusion" is performed for low_res image.
    # Note that if a model doesn't have low_res input (i.e. base model), this option is simply ignored.
    noisy_low_res: bool = True 

    # loss
    loss_scaling: float = 1.0
    lr: float = 1e-4
    clip_grad: Union[None, float] = None
    lr_scheduler: Union[None, str] = None

    # for classifier-free guidance
    cond_drop_rate: float = 0.1

# expose config to enable loading from yaml file
from .utils import register_config
register_config(name="train", node=TrainConfig)
