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

from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()


def register_config(name, node, group="proto_configs"):
    """
    Registers config node which is defined as dataclass to enable loading from yaml file.
    The registerd config can be accessed by "/{group}/{name}" like the example below.
    example: 
        ``` hoge.py    
        @dataclass
        def HogeConfig:
            a: int = 1
            b: int = MISSING

        register_config(config_name="hoge", node=HogeConfig, group="proto_configs")
        ```

        ``` hoge.yaml 
        defaults:
            # you can load default values defined by HogeConfig here
            - /proto_configs/hoge@_here_

        # a is loaded from HogeConfig and the value is 1.

        b: 3  # you can overwrite values 
        ```
    """

    cs.store(group=group, name=name, node=node)
