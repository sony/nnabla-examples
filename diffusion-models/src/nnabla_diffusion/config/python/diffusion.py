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

# Diffusion definition


@dataclass
class DiffusionConfig:
    beta_strategy: str = "linear"
    max_timesteps: int = 1000
    t_start: int = "${diffusion.max_timesteps}"
    respacing_step: int = 1
    model_var_type: str = "${model.model_var_type}"


# expose config to enable loading from yaml file
from .utils import register_config
register_config(name="diffusion", node=DiffusionConfig)
