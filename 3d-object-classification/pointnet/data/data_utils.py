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

from typing import Any, List
import numpy as np
import pickle


def normalize_point_cloud(data: np.ndarray) -> np.ndarray:
    """normalize point cloud
    Args:
        data (np.ndarray): shape(batch, num_points, dim)

    Returns:
        np.ndarray: normalized point cloud
    """
    transformed_data = data.copy()
    centroid = np.mean(transformed_data, axis=1)
    transformed_data = transformed_data - centroid
    max_vector = np.max(np.sqrt(np.sum(transformed_data ** 2, axis=2)), axis=1)
    transformed_data = transformed_data / max_vector
    return transformed_data


def load_txt_file(file_path: str) -> List[str]:
    with open(file_path) as f:
        file_lines = [line.rstrip() for line in f]
    return file_lines


def save_as_pickle(obj: Any, file_path: str) -> None:
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def load_from_pickle(file_path: str) -> Any:
    with open(file_path, "rb") as f:
        obj = pickle.load(f)
    return obj
