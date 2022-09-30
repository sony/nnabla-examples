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

from dataclasses import dataclass, field
from typing import Any, List, Union

from omegaconf import MISSING, OmegaConf

# todo: using config name to access other config group is unsafe

# Model Definition

# set resolvers
def get_output_channels(input_channels, var_type) -> int:
    # calc output channels from input and vartype
    from diffusion_model.diffusion import is_learn_sigma
    if is_learn_sigma(var_type):
        return input_channels * 2

    return input_channels


OmegaConf.register_new_resolver("get_oc", get_output_channels)


def get_image_shape(image_size, input_channels, channel_last):
    if image_size is None:
        return None

    # return input image_shape from input and vartype
    if channel_last:
        return image_size + (input_channels, )

    return (input_channels, ) + image_size


OmegaConf.register_new_resolver("get_is", get_image_shape)


def is_noisy_lr(low_res_size, noisy_lr):
    if low_res_size is None:
        return False

    return noisy_lr


OmegaConf.register_new_resolver("is_noisy_lr", is_noisy_lr)


def get_text_emb_shape(max_length, emb_dim, channel_last):
    if max_length == 0 or emb_dim == 0:
        return None

    if channel_last:
        return (max_length, emb_dim)

    return (emb_dim, max_length)


OmegaConf.register_new_resolver("get_tes", get_text_emb_shape)


@dataclass
class ModelConfig:
    # input
    input_channels: int = 3
    image_size: List[int] = MISSING
    image_shape: Union[None, List[int]] = \
        "${get_is:${model.image_size},${model.input_channels},${model.channel_last}}"

    low_res_size: Union[None, List[int]] = None
    low_res_shape: Union[None, List[int]] = \
        "${get_is:${model.low_res_size},${model.input_channels},${model.channel_last}}"
    noisy_low_res: bool = "${is_noisy_lr:${model.low_res_size},${train.noisy_low_res}}"

    # arch.
    arch: str = "unet"
    scale_shift_norm: bool = True
    resblock_resample: bool = False
    resblock_rescale_skip: bool = False
    num_res_blocks: Any = MISSING
    channel_mult: List[int] = MISSING
    base_channels: int = 128
    dropout: float = 0.
    channel_last: bool = True
    conv_resample: bool = True
    use_mixed_precision: bool = True

    # attention
    attention_type: str = "self_attention"
    attention_resolutions: List[int] = field(
        default_factory=lambda: [8, 16, 32])
    num_attention_head_channels: Union[None, int] = 64
    num_attention_heads: Union[None, int] = None
    # num_attention_head_channels is prioritized over num_attention_heads if both are specified.

    # class condition
    class_cond: bool = False
    class_cond_emb_type: str = "simple"
    num_classes: int = "${dataset.num_classes}"

    # text condition
    text_cond: bool = False
    text_cond_emb_type: str = "ln_mlp"
    max_text_length: int = "${dataset.max_text_length}"
    text_emb_dims: int = "${dataset.text_emb_dims}"
    text_emb_shape: Union[None, List[int]
                          ] = "${get_tes:${dataset.max_text_length},${dataset.text_emb_dims},${model.channel_last}}"

    # output
    model_var_type: str = "learned_range"
    output_channels: int = \
        "${get_oc:${model.input_channels},${model.model_var_type}}"


# expose config to enable loading from yaml file
from .utils import register_config
register_config(name="model", node=ModelConfig)
