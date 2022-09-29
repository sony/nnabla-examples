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
from typing import List, Union, Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf

# todo: using config name to access other config group is unsafe

# nnabla runtime definition


@dataclass
class RuntimeConfig:
    type_config: str = "half"
    device_id: str = "0"

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

# Model Definition

# set resolvers


def get_output_channels(input_channels, var_type) -> int:
    # calc output channels from input and vartype
    from diffusion import is_learn_sigma
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
    text_emb_shape: Union[None, List[int]] = "${get_tes:${dataset.max_text_length},${dataset.text_emb_dims},${model.channel_last}}"

    # output
    model_var_type: str = "learned_range"
    output_channels: int = \
        "${get_oc:${model.input_channels},${model.model_var_type}}"


# Diffusion definition
@dataclass
class DiffusionConfig:
    beta_strategy: str = "linear"
    max_timesteps: int = 1000
    t_start: int = "${diffusion.max_timesteps}"
    respacing_step: int = 1
    model_var_type: str = "${model.model_var_type}"


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


@dataclass
class GenerateConfig:
    # load
    config: str = MISSING
    h5: str = MISSING

    # generation configuration
    ema: bool = True
    ddim: bool = False
    # todo: add seed to fix ddim sampling
    ode_solver: Union[None, str] = None
    # currently supporting "plms" and "dpm2"
    samples: int = 1024
    batch_size: int = 36

    # for refinement
    respacing_step: int = 4
    t_start: Union[None, int] = None

    # nstep
    base_samples_dir: Union[None, str] = None

    # class cond
    gen_class_id: Union[None, int] = None
    classifier_free_guidance_weight: Union[None, float] = None

    # for SDEidt
    x_start_path: Union[None, str] = None

    # dump
    output_dir: str = "./outs"
    tiled: bool = True
    save_xstart: bool = False

# Config for training scripts


@dataclass
class TrainScriptConfig:
    runtime: RuntimeConfig = MISSING
    dataset: DatasetConfig = MISSING
    model: ModelConfig = MISSING
    diffusion: DiffusionConfig = MISSING
    train: TrainConfig = MISSING

# Config for generation scripts


@dataclass
class GenScriptConfig:
    runtime: RuntimeConfig = MISSING
    generate: GenerateConfig = MISSING

# for loading config file


@dataclass
class LoadedConfig:
    diffusion: DiffusionConfig
    model: ModelConfig


def load_saved_conf(config_path: str) -> LoadedConfig:
    import os
    assert os.path.exists(config_path), \
        f"config file {config_path} is not found."

    base_conf = OmegaConf.structured(LoadedConfig)
    loaded_conf = OmegaConf.load(config_path)

    assert hasattr(loaded_conf, "diffusion"), \
        f"config file must have a `diffusion` entry."
    assert hasattr(loaded_conf, "model"), \
        f"config file must have a `model` entry."

    return OmegaConf.merge(base_conf, loaded_conf)

# config register


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(
        group="proto_configs",
        name="dataset",
        node=DatasetConfig
    )

    cs.store(
        group="proto_configs",
        name="model",
        node=ModelConfig
    )

    cs.store(
        group="proto_configs",
        name="diffusion",
        node=DiffusionConfig
    )

    cs.store(
        group="proto_configs",
        name="train",
        node=TrainConfig
    )

    cs.store(
        group="proto_configs",
        name="generate",
        node=GenerateConfig
    )
