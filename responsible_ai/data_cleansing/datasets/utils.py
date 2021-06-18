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


def get_filename_to_download(output_dir: str, data_uri: str):
    if output_dir is None:
        output_file = None
    else:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(output_dir, data_uri.split('/')[-1])
    return output_file


def save_list_to_csv(csv_data, csv_path, csv_file_name):
    with open(os.path.join(csv_path, csv_file_name), 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(csv_data)


def split_data_into_train_val(data_list, val_size=10000, seed=0):
    n = len(data_list) - 1  # forst row is title
    np.random.seed(seed)
    idx_val = np.random.permutation(n)[:val_size]
    idx_train = np.setdiff1d(np.arange(n), idx_val)
    data = data_list[1:]
    train_data = [data_list[0]] + [data[i] for i in idx_train]
    val_data = [data_list[0]] + [data[i] for i in idx_val]
    return train_data, val_data


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
