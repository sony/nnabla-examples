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

from omegaconf import MISSING, OmegaConf

from .dataset import DatasetConfig
from .diffusion import DiffusionConfig
from .generate import GenerateConfig
from .model import ModelConfig
from .train import TrainConfig
from .ui import InferenceServerConfig


@dataclass
class RuntimeConfig:
    device_id: str = "0"


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


# Config for inference server
@dataclass
class InferenceServerScriptConfig:
    runtime: RuntimeConfig = MISSING
    server: InferenceServerConfig = MISSING


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
