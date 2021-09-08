# Copyright 2021 Sony Corporation.
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

import os
import csv
import numpy as np
from nnabla.ext_utils import get_extension_context
from nnabla.utils.load import load
from shutil import rmtree


def delete_file(file_name):
    if os.path.isfile(file_name):
        os.remove(file_name)


def delete_dir(dir_name, keyword='sgd_infl_results'):
    if os.path.isdir(dir_name):
        if keyword in dir_name:
            rmtree(dir_name)


def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def get_context(device_id):
    # for cli app use
    try:
        context = 'cudnn'
        ctx = get_extension_context(context, device_id=device_id)
    except ModuleNotFoundError:
        context = 'cpu'
        ctx = get_extension_context(context, device_id=device_id)
    # for nnc use
    config_filename = 'net.nntxt'
    if os.path.isfile(config_filename):
        config_info = load([config_filename])
        ctx = config_info.global_config.default_context

    return ctx


def get_indices(n, seed):
    np.random.seed(seed)
    idx = np.random.permutation(n)
    return idx


def save_to_csv(filename, header, list_to_save, data_type):
    with open(filename, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(header)
        writer.writerows(np.array([tuple(row)
                                   for row in list_to_save], dtype=data_type))


def read_csv(filename):
    with open(filename) as f:
        reader = csv.reader(f)
        ret = [s for s in reader]
    return ret
