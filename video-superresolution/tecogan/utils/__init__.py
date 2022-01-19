# Copyright 2020,2021 Sony Corporation.
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
from __future__ import absolute_import
import os
import sys
common_utils_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'utils'))
print(common_utils_path)
sys.path.append(common_utils_path)
from neu.yaml_wrapper import read_yaml
from neu.misc import AttrDict
from neu.comm import CommunicatorWrapper
from neu.variable_utils import set_persistent_all
from neu.checkpoint_util import save_checkpoint, load_checkpoint
