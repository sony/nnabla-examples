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
from typing import Any, Dict, Optional, Tuple, Iterable
import numpy as np
from tqdm import tqdm
import pickle
import json

from nnabla.utils.data_iterator import DataIterator, data_iterator
from nnabla.utils.data_source import DataSource
from nnabla.logger import logger

CLASS_NAME_TO_SEGMENTATION_ID_DICT = {
    "Earphone": [16, 17, 18],
    "Motorbike": [30, 31, 32, 33, 34, 35],
    "Rocket": [41, 42, 43],
    "Car": [8, 9, 10, 11],
    "Laptop": [28, 29],
    "Cap": [6, 7],
    "Skateboard": [44, 45, 46],
    "Mug": [36, 37],
    "Guitar": [19, 20, 21],
    "Bag": [4, 5],
    "Lamp": [24, 25, 26, 27],
    "Table": [47, 48, 49],
    "Airplane": [0, 1, 2, 3],
    "Pistol": [38, 39, 40],
    "Chair": [12, 13, 14, 15],
    "Knife": [22, 23],
}

SEGMENTATION_ID_TO_CLASS_NAME_DICT = {}

for class_name in CLASS_NAME_TO_SEGMENTATION_ID_DICT.keys():
    for seg_id in CLASS_NAME_TO_SEGMENTATION_ID_DICT[class_name]:
        SEGMENTATION_ID_TO_CLASS_NAME_DICT[seg_id] = class_name


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


def shift_point_cloud(data: np.ndarray, shift_range: float = 0.1) -> np.ndarray:
    """shift point cloud

    Args:
        data (np.ndarray): data
        shift_range (float): shift range

    Returns:
        np.ndarray: shifted point cloud
    """
    transformed_data = data.copy()
    batch_size, num_points, _ = transformed_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (batch_size, 1, 3))
    transformed_data += shifts
    return transformed_data


def scale_point_cloud(data: np.ndarray, scale_low: float = 0.8, scale_high: float = 1.25) -> np.ndarray:
    """scale point cloud

    Args:
        data (np.ndarray): data
        scale_low (float): low scale value
        scale_high (float): high scale value

    Returns:
        np.ndarray: scaled point cloud
    """
    transformed_data = data.copy()
    batch_size, num_points, _ = transformed_data.shape
    scales = np.random.uniform(scale_low, scale_high, (batch_size, 1, 1))
    transformed_data *= scales
    return transformed_data


def save_as_pickle(obj: Any, file_path: str) -> None:
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def load_from_pickle(file_path: str) -> Any:
    with open(file_path, "rb") as f:
        obj = pickle.load(f)
    return obj


def load_txt_as_np_array(
    data_paths: Iterable[Tuple[str, str]], classes_dict: Dict[str, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    point_cloud_data = []
    segmentation_data = []
    label_data = []

    for data_path in tqdm(data_paths, total=len(data_paths)):
        class_name, point_cloud_data_path = data_path
        # point cloud
        point_cloud_with_seg_label = np.loadtxt(
            point_cloud_data_path).astype(np.float32)
        point_cloud = point_cloud_with_seg_label[:, :6]
        segmentation = point_cloud_with_seg_label[:, -1]
        point_cloud_data.append(point_cloud)
        segmentation_data.append(segmentation)
        # label
        label = int(classes_dict[class_name])
        label_data.append(label)

    return point_cloud_data, label_data, segmentation_data


class ShapenetCorePartannoSegDataset(DataSource):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        train: bool,
        shuffle: bool,
        num_points: int,
        normalize: bool,
        scale: bool,
        shift: bool,
        with_normal: bool,
        rng: Optional[int] = None,
    ) -> None:
        super().__init__(shuffle=shuffle, rng=rng)
        self._shuffle = shuffle
        self._train = train
        self._batch_size = batch_size
        self._num_points = num_points

        self._normalize = normalize
        self._scale = scale
        self._shift = shift
        self._with_normal = with_normal

        if self._train:
            processed_data_path = os.path.join(
                data_dir, "train_shapenet_partanno_segmentation.pkl")
        else:
            processed_data_path = os.path.join(
                data_dir, "test_shapenet_partanno_segmentation.pkl")

        category_file_path = os.path.join(
            data_dir, "synsetoffset2category.txt")

        self._category = {}
        with open(category_file_path, "r") as f:
            for line in f:
                lines = line.strip().split()
                self._category[lines[0]] = lines[1]
        self._classes_dict = dict(
            zip(self._category, range(len(self._category))))

        with open(os.path.join(data_dir, "train_test_split", "shuffled_train_file_list.json"), "r") as f:
            train_ids = set([str(d.split("/")[2]) for d in json.load(f)])
        with open(os.path.join(data_dir, "train_test_split", "shuffled_val_file_list.json"), "r") as f:
            val_ids = set([str(d.split("/")[2]) for d in json.load(f)])
        with open(os.path.join(data_dir, "train_test_split", "shuffled_test_file_list.json"), "r") as f:
            test_ids = set([str(d.split("/")[2]) for d in json.load(f)])

        meta_file_list = {}
        for category_name, category_dir_id in self._category.items():
            meta_file_list[category_name] = []
            dir_point = os.path.join(data_dir, category_dir_id)
            file_list = sorted(os.listdir(dir_point))

            if train:
                file_list = [
                    file_name
                    for file_name in file_list
                    if ((file_name[0:-4] in train_ids) or (file_name[0:-4] in val_ids))
                ]
            else:
                file_list = [
                    file_name for file_name in file_list if file_name[0:-4] in test_ids]

            for file_name in file_list:
                token = os.path.splitext(os.path.basename(file_name))[0]
                meta_file_list[category_name].append(
                    os.path.join(dir_point, token + ".txt"))

        self._all_data_path = []
        for category_name, category_dir_id in self._category.items():
            for file_name in meta_file_list[category_name]:
                self._all_data_path.append((category_name, file_name))

        if not os.path.exists(processed_data_path):
            logger.info("Load from original datasets ...")
            self._point_clouds, self._labels, self._segmentations = load_txt_as_np_array(
                self._all_data_path, self._classes_dict
            )
            logger.info(f"Saving data as pkl ... to {processed_data_path}")
            save_as_pickle((self._point_clouds, self._labels,
                           self._segmentations), processed_data_path)
        else:
            logger.info("Load from processed datasets ...")
            processed_data = load_from_pickle(processed_data_path)
            self._point_clouds, self._labels, self._segmentations = processed_data

        self._size = len(self._point_clouds)
        self._variables = ("point_cloud", "category", "segmentation")
        self.rng = np.random.RandomState(rng)
        self.reset()

    def _get_data(self, position):
        index = self._indexes[position]
        point_cloud = np.array(self._point_clouds[index], dtype=np.float32)[
                               np.newaxis, :, :]
        segmentation = np.array(self._segmentations[index], dtype=np.int32)
        category = np.array([self._labels[index]], dtype=np.int32)

        if self._scale:
            point_cloud[:, :, :3] = scale_point_cloud(point_cloud[:, :, :3])

        if self._shift:
            point_cloud[:, :, :3] = shift_point_cloud(point_cloud[:, :, :3])

        if self._normalize:
            point_cloud[:, :, :3] = normalize_point_cloud(
                point_cloud[:, :, :3])

        if not self._with_normal:
            point_cloud = point_cloud[:, :, :3]

        choice_idx = np.random.choice(
            len(segmentation), self._num_points, replace=True)

        point_cloud = point_cloud[:, choice_idx, :]
        segmentation = segmentation[choice_idx]

        return (np.squeeze(point_cloud), category, segmentation)

    def reset(self):
        if self._shuffle:
            self._indexes = self.rng.permutation(self._size)
        else:
            self._indexes = np.arange(self._size)
        super(ShapenetCorePartannoSegDataset, self).reset()


def data_iterator_shapenet_partanno_segmentation(
    data_dir: str,
    batch_size: int,
    train: bool,
    shuffle: bool,
    num_points: int,
    normalize: bool,
    with_normal: bool,
    scale: bool,
    shift: bool,
    use_thread: bool = True,
    stop_exhausted: bool = False,
    with_memory_cache: bool = False,
    with_file_cache: bool = False,
    rng: Optional[int] = None,
) -> DataIterator:
    dataset = ShapenetCorePartannoSegDataset(
        data_dir, batch_size, train, shuffle, num_points, normalize, scale, shift, with_normal
    )
    return data_iterator(
        dataset,
        batch_size,
        rng=rng,
        use_thread=use_thread,
        stop_exhausted=stop_exhausted,
        with_memory_cache=with_memory_cache,
        with_file_cache=with_file_cache,
    )
