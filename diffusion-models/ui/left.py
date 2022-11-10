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
import gradio as gr

from config import LoadedConfig


@dataclass
class LeftBlocks():
    # inference configurations
    respacing_step: gr.Slider
    sampler: gr.Slider

    # for class conditional model
    class_id: Union[None, gr.Dropdown]
    classifier_free_guidance_weight: Union[None, gr.Dropdown]

    # for upsampler
    lowres_noise_level: Union[None, gr.Dropdown]


def respacing_step_block(max_timestep: int,
                         prefix: str) -> gr.Slider:
    return gr.Slider(minimum=0,
                     maximum=max_timestep,
                     value=max_timestep // 50,  # default value
                     step=1,
                     label=f"[{prefix}] Respacing step")


def sampler_block(prefix: str) -> gr.Dropdown:
    return gr.Dropdown(choices=["ddpm", "ddim", "plms", "dpm2"],
                       value="ddpm",
                       label=f"[{prefix}] sampler")


def class_id_block(is_class_cond: bool,
                   num_classes: int,
                   prefix: str) -> gr.Dropdown:

    # todo: show class name rather than id
    return gr.Dropdown(choices=list(range(0, num_classes)),
                       value=0,
                       visible=is_class_cond,
                       label=f"[{prefix}] class id")


def cfguide_block(is_class_cond: bool,
                  prefix: str) -> gr.Slider:

    return gr.Slider(minimum=0,
                     maximum=100,
                     value=0.1,
                     step=0.1,
                     visible=is_class_cond,
                     label=f"[{prefix}] classifier-free guidance weight")


def lowres_noise_level_block(is_upsampler: bool,
                             max_timestep: int,
                             prefix: str):

    return gr.Slider(minimum=0,
                     maximum=max_timestep,
                     value=0,
                     step=1,
                     visible=is_upsampler,
                     label=f"[{prefix}] noise timestep for low-resolution image")


def create_left_column_from_config(conf: LoadedConfig, prefix: str) -> LeftBlocks:
    # respacing
    block1 = respacing_step_block(max_timestep=conf.diffusion.max_timesteps,
                                  prefix=prefix)

    # sampler
    block2 = sampler_block(prefix)

    # class id
    block3 = class_id_block(is_class_cond=conf.model.class_cond,
                            num_classes=conf.model.num_classes,
                            prefix=prefix)

    # classifier-free guidance weight
    block4 = cfguide_block(is_class_cond=conf.model.class_cond,
                           prefix=prefix)

    # lowres noise loevel
    block5 = lowres_noise_level_block(is_upsampler=conf.model.low_res_size is not None,
                                      max_timestep=conf.diffusion.max_timesteps,
                                      prefix=prefix)

    return LeftBlocks(respacing_step=block1,
                      sampler=block2,
                      class_id=block3,
                      classifier_free_guidance_weight=block4,
                      lowres_noise_level=block5)
