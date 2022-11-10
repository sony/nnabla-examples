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
from omegaconf import MISSING
from typing import Optional

# configuration


@dataclass
class InferenceServerConfig:
    port: int = 50000

    # base model
    base_h5_path: str = MISSING
    base_conf_path: str = MISSING

    # upsampler 1
    up1_h5_path: Optional[str] = None
    up1_conf_path: Optional[str] = None


# expose config to enable loading from yaml file
from .utils import register_config
register_config(name="inference_server", node=InferenceServerConfig)
