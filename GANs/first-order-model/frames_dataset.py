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
import glob
import random
import numpy as np
from imageio import mimread

import nnabla.logger as logger

from nnabla.utils.data_iterator import data_iterator
from nnabla.utils.data_source import DataSource
from nnabla.utils.image_utils import imread


def read_video(name, frame_shape):
    """
        note that this function assumes that data (images or a video)
        is stored as RGB format.
    """

    if os.path.isdir(name):
        frames = sorted(os.listdir(name))
        num_frames = len(frames)
        video_array = np.array(
             [imread(os.path.join(name, frames[idx])) / 255. for idx in range(num_frames)])

    elif name.lower().endswith('.gif') or name.lower().endswith('.mp4') or name.lower().endswith('.mov'):
        video = np.array(mimread(name, memtest=False,
                                 size=tuple(frame_shape[:2])))
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array = video / 255.
    else:
        raise Exception("Unknown file extensions  %s" % name)

    return video_array


class FramesDataSource(DataSource):
    def __init__(self, root_dir, frame_shape=(256, 256, 3),
                 id_sampling=False, is_train=True,
                 random_seed=0,
                 augmentation_params=None,
                 shuffle=True):
        super(FramesDataSource, self).__init__()

        self.root_dir = root_dir
        self.videos = os.listdir(root_dir)
        self.frame_shape = tuple(frame_shape)
        self.id_sampling = id_sampling
        self._shuffle = shuffle

        if os.path.exists(os.path.join(root_dir, 'train')):
            assert os.path.exists(os.path.join(root_dir, 'test'))
            logger.info("Use predefined train-test split.")
            if id_sampling:
                train_videos = {os.path.basename(video).split('#')[0] for video in
                                os.listdir(os.path.join(root_dir, 'train'))}
                train_videos = list(train_videos)
            else:
                train_videos = os.listdir(os.path.join(root_dir, 'train'))
            test_videos = os.listdir(os.path.join(root_dir, 'test'))

            if is_train:
                self.root_dir = os.path.join(self.root_dir, 'train')
            else:
                self.root_dir = os.path.join(self.root_dir, 'test')

        else:
            logger.info("Use random train-test split.")
            random.shuffle(self.videos)
            num_test_samples = int(len(self.videos) * 0.2)
            train_videos, test_videos = self.videos[num_test_samples:], self.videos[:num_test_samples]

        if is_train:
            self.videos = train_videos
        else:
            self.videos = test_videos

        self.is_train = is_train

        if self.is_train:
            self.transform = True
        else:
            self.transform = None

        logger.info(f'using data in {self.root_dir}')

        # requirement
        self._size = len(self.videos)
        if self.is_train:
            self._variables = ('driving', 'source')
        else:
            self._variables = ('video', 'name')
        self.reset()

    def _get_data(self, position):
        idx = self._indexes[position]

        if self.is_train and self.id_sampling:
            name = self.videos[idx]
            path = np.random.choice(
                glob.glob(os.path.join(self.root_dir, name + '*.mp4')))
            path = str(path)
        else:
            name = self.videos[idx]
            path = os.path.join(self.root_dir, name)

        if self.is_train and os.path.isdir(path):
            frames = os.listdir(path)
            num_frames = len(frames)
            frame_idx = np.sort(np.random.choice(
                num_frames, replace=True, size=2))
            video_array = [
                imread(os.path.join(path, frames[idx])) / 255.0 for idx in frame_idx]

        else:
            video_array = read_video(path, frame_shape=self.frame_shape)
            num_frames = len(video_array)
            if self.is_train:
                frame_idx = np.sort(np.random.choice(
                    num_frames, replace=True, size=2))
            else:
                frame_idx = range(num_frames)
            video_array = video_array[frame_idx]

        if self.transform is not None:
            if random.random() < 0.5:
                video_array = video_array[::-1]
            if random.random() < 0.5:
                video_array = [np.fliplr(img) for img in video_array]

        out = {}
        if self.is_train:
            source = np.array(video_array[0], dtype='float32')
            driving = np.array(video_array[1], dtype='float32')

            out['driving'] = driving.transpose((2, 0, 1))
            out['source'] = source.transpose((2, 0, 1))
        else:
            video = np.array(video_array, dtype='float32')
            out['video'] = video.transpose((3, 0, 1, 2))

        if self.is_train:
            return out["driving"], out["source"]
        else:
            return out["video"], out["name"]

    def reset(self):
        # reset method initialize self._indexes
        if self._shuffle:
            self._indexes = np.arange(self._size)
            np.random.shuffle(self._indexes)
        else:
            self._indexes = np.arange(self._size)
        super(FramesDataSource, self).reset()


def frame_data_iterator(root_dir, frame_shape=(256, 256, 3), id_sampling=False,
                        is_train=True, random_seed=0,
                        augmentation_params=None, batch_size=1, shuffle=True,
                        with_memory_cache=False, with_file_cache=False):
    return data_iterator(FramesDataSource(root_dir=root_dir,
                                          frame_shape=frame_shape,
                                          id_sampling=id_sampling,
                                          is_train=is_train,
                                          random_seed=random_seed,
                                          augmentation_params=augmentation_params,
                                          shuffle=shuffle),
                         batch_size=batch_size,
                         rng=random_seed,
                         with_memory_cache=with_memory_cache,
                         with_file_cache=with_file_cache)
