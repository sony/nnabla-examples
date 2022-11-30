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

import numpy as np
import webdataset as wds
from config import DatasetConfig
from neu.comm import CommunicatorWrapper
from nnabla.logger import logger
from nnabla.utils.data_iterator import data_iterator
from nnabla.utils.data_source import DataSource

from .common import resize_center_crop, resize_random_crop


class WebDatasetDataSourceLocal(DataSource):
    '''
    DataSource for webdataset format.

    Assumes all tar files are already donwnloaded on the same server and accecible as a file. 
    '''

    def __init__(self,
                 tar_files,
                 conf: DatasetConfig,
                 rng=None):
        super(WebDatasetDataSourceLocal, self).__init__(
            shuffle=conf.shuffle_dataset, rng=rng)

        shuffle_size = 1
        if conf.shuffle_dataset:
            shuffle_size = 10000

        self.dataset = iter(wds.DataPipeline(
           wds.ResampledShards(tar_files),
           wds.tarfile_to_samples(),
           wds.shuffle(shuffle_size),
           wds.decode("rgb"),
           wds.to_tuple("jpg", "json", "npz")
        ))

        self._size = -1
        self._variables = ("image", "caption", "t5_emb")
        self.im_size = conf.image_size
        self.fix_aspect_ratio = conf.fix_aspect_ratio
        self.random_crop = conf.random_crop
        self.channel_last = conf.channel_last
        self.max_text_length = conf.max_text_length

    def _get_data(self, position):
        # Note that position is not used in this data source
        data = next(self.dataset)

        image = data[0]  # numpy whose value is between [0, 1], channel last
        cap = data[1]["caption"]  # str
        emb = data[2]["t5_emb"]  # (length, 1024)

        # rescale pixel intensity to [-1, 1]
        image = 2 * image - 1

        # resize image to align config
        if self.random_crop:
            image = resize_random_crop(
                image, size=self.im_size[0], channel_first=False)
        else:
            image = resize_center_crop(
                image, size=self.im_size[0], channel_first=False)

        # padding text sequence
        # Truncate emb if it's longer than max length.
        emb = emb[:self.max_text_length]
        # padding to max length
        emb = np.pad(emb, ((0, self.max_text_length - emb.shape[0]), (0, 0)))

        if not self.channel_last:
            # channel last -> first
            image = np.transpose(image, (2, 0, 1))
            emb = np.transpose(emb, (1, 0))

        return (image, cap, emb)


TAR_FILES = {
    "400m": "{00000..41407}.tar",
}


def Laion400mDataIterator(conf: DatasetConfig, comm: CommunicatorWrapper):
    # set worker info to avoid loading pytorch in webdataset
    os.environ["RANK"] = str(comm.rank)
    os.environ["WORLD_SIZE"] = str(comm.n_procs)
    os.environ["WORKER"] = str(comm.rank)
    os.environ["NUM_WORKERS"] = str(comm.n_procs)

    # create datasource
    # tar_files = os.path.join(conf.dataset_root_dir, TAR_FILES["400m"])

    tar_list_path = os.path.join(conf.data_dir, "tarlist.txt")
    with open(tar_list_path, "r") as f:
        tar_files = [x.strip() for x in f.readlines()]
    logger.info(
        f"[LAION400mDataIterator] {len(tar_files)} tarfiles are found.")

    ds = WebDatasetDataSourceLocal(tar_files, conf)

    return data_iterator(ds,
                         conf.batch_size,
                         with_memory_cache=False,
                         use_thread=True,
                         with_file_cache=False)
