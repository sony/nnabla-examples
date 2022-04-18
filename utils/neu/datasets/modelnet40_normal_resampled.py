# Copyright 2022 Sony Group Corporation.
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
from typing import Any, Dict, List, Optional, Tuple, Iterable
import numpy as np
from tqdm import tqdm
import pickle

from nnabla.utils.data_iterator import DataIterator, data_iterator
from nnabla.utils.data_source import DataSource
from nnabla.logger import logger


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
    max_vector = np.max(np.sqrt(np.sum(transformed_data**2, axis=2)), axis=1)
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


def load_dataset_path_file(data_dir: str, txt_file_name: str) -> List[str]:
    data_lists = load_txt_file(os.path.join(data_dir, txt_file_name))
    shape_names = ["_".join(x.split("_")[0:-1]) for x in data_lists]
    data_paths = [
        (shape_names[i], os.path.join(data_dir, shape_names[i], data_lists[i]) + ".txt") for i in range(len(data_lists))
    ]
    return data_paths


def load_txt_as_np_array(
    data_paths: Iterable[Tuple[str, str]], num_points: int, classes_dict: Dict[str, int]
) -> Tuple[np.ndarray, np.ndarray]:
    points_data = []
    label_data = []

    for data_path in tqdm(data_paths, total=len(data_paths)):
        class_name, point_cloud_data_path = data_path
        # point cloud
        point_cloud = np.loadtxt(
            point_cloud_data_path, delimiter=",").astype(np.float32)
        point_cloud = point_cloud[:num_points, :]
        points_data.append(point_cloud)
        # label
        label = int(classes_dict[class_name])
        label_data.append(label)

    return np.array(points_data, dtype=np.float32), np.array(label_data, dtype=np.int32)


class ModelNet40NormalResampledDataset(DataSource):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        train: bool,
        shuffle: bool,
        num_points: int,
        normalize: bool,
        with_normal: bool,
        rng: Optional[int] = None,
    ) -> None:
        super().__init__(shuffle=shuffle, rng=rng)
        self._shuffle = shuffle
        self._train = train
        self._batch_size = batch_size
        self._normalize = normalize
        self._with_normal = with_normal

        if self._train:
            processed_data_path = os.path.join(
                data_dir, "train_modelnet40_normal_resampled.pkl")
        else:
            processed_data_path = os.path.join(
                data_dir, "test_modelnet40_normal_resampled.pkl")

        self._shape_names = load_txt_file(
            os.path.join(data_dir, "modelnet40_shape_names.txt"))
        self._classes_dict = dict(
            zip(self._shape_names, range(len(self._shape_names))))

        if not os.path.exists(processed_data_path):
            logger.info("Load from original datasets ...")
            txt_file_name = "modelnet40_train.txt" if train else "modelnet40_test.txt"
            data_paths = load_dataset_path_file(data_dir, txt_file_name)
            self._point_clouds, self._labels = load_txt_as_np_array(
                data_paths, num_points, self._classes_dict)
            logger.info(f"Saving data as pkl ... to {processed_data_path}")
            save_as_pickle((self._point_clouds, self._labels),
                           processed_data_path)
        else:
            logger.info("Load from processed datasets ...")
            processed_data = load_from_pickle(processed_data_path)
            self._point_clouds, self._labels = processed_data

        self._size = len(self._point_clouds)
        self._variables = ("point_cloud", "label")
        self.rng = np.random.RandomState(rng)
        self.reset()

    def _get_data(self, position):
        index = self._indexes[position]
        point_cloud = np.array(self._point_clouds[index], dtype=np.float32)[
                               np.newaxis, :, :]
        label = np.array([self._labels[index]], dtype=np.int32)

        if self._normalize:
            point_cloud[:, :, :3] = normalize_point_cloud(
                point_cloud[:, :, :3])

        if not self._with_normal:
            point_cloud = point_cloud[:, :, :3]

        return (np.squeeze(point_cloud), label)

    def reset(self) -> None:
        if self._shuffle:
            self._indexes = self.rng.permutation(self._size)
        else:
            self._indexes = np.arange(self._size)
        super(ModelNet40NormalResampledDataset, self).reset()


def data_iterator_modelnet40_normal_resampled(
    data_dir: str,
    batch_size: int,
    train: bool,
    shuffle: bool,
    num_points: int,
    normalize: bool,
    with_normal: bool,
    use_thread: bool = True,
    stop_exhausted: bool = False,
    with_memory_cache: bool = False,
    with_file_cache: bool = False,
    rng: Optional[int] = None,
) -> DataIterator:
    dataset = ModelNet40NormalResampledDataset(
        data_dir,
        batch_size,
        train,
        shuffle,
        num_points,
        normalize,
        with_normal,
    )
    return data_iterator(
        dataset,
        batch_size,
        rng=rng,
        use_thread=use_thread,
        with_memory_cache=with_memory_cache,
        with_file_cache=with_file_cache,
        stop_exhausted=stop_exhausted,
    )
