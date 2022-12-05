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
class GenerateConfig:
    # load
    config: str = MISSING
    h5: str = MISSING

    # generation configuration
    ema: bool = True
    # currently supporting ["ddpm", "ddim", "ddim_rev", "plms", "dpm2"]
    sampler: str = "ddpm"
    samples: int = 1024
    batch_size: int = 32

    # for refinement
    respacing_step: int = 4
    t_start: Union[None, int] = None

    # nstep
    lowres_aug_timestep: Union[None, int] = None
    base_samples_dir: Union[None, str] = None

    # condition
    classifier_free_guidance_weight: Union[None, float] = None
    # class cond
    gen_class_id: Union[None, int] = None
    # text cond
    text: Union[None, str] = None

    # Input image for SDEidt, etc
    x_start: Union[None, str] = None

    # dump
    output_dir: str = "./outs"
    tiled: bool = True
    save_xstart: bool = False


# expose config to enable loading from yaml file
from .utils import register_config
register_config(name="generate", node=GenerateConfig)
