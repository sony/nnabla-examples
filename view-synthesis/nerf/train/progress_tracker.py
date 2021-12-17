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

import nnabla as nn
import nnabla.monitor as nm


class MonitorManager(object):
    """
        input_dict = {str1: Variable1, str2: Variable2, ... }
    """

    def __init__(self, train_key2var_dict, test_key2var_dict, path):

        monitor = nm.Monitor(path)
        self.monitor_dict = dict()

        for k, v in train_key2var_dict.items():
            self.monitor_dict[k] = nm.MonitorSeries(
                k, monitor, interval=1)

        for k, v in test_key2var_dict.items():
            self.monitor_dict[k] = nm.MonitorSeries(
                k, monitor, interval=1)

    def add(self, i, var_dict):
        for k, v in var_dict.items():
            self.monitor_dict[k].add(i, v)
