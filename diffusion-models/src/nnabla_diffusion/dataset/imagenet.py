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
import pathlib

import click
import numpy as np
from neu.comm import CommunicatorWrapper
from neu.datasets import _get_sliced_data_source
from nnabla import logger
from nnabla.utils.data_iterator import data_iterator
from nnabla_diffusion.config import DatasetConfig

from .common import SimpleDatasource

# ImagenetDataIterator uses label_wordnetid.csv, label_words.csv, and validation_data_label.txt
DEFAULT_RESOURCE_DIR = os.path.join(os.path.dirname(__file__),  # nnabla-examples/diffusion-models/dataset
                                    "..",  # nnabla-examples/diffusion-models
                                    "..",  # nnabla-examples
                                    "image-classification/imagenet"  # nnabla-examples/image-classification/imagenet
                                    )


def _info(msg):
    prefix = "[ImagenetDataIterator]"
    logger.info(f"{prefix} {msg}")


def ImagenetDataIterator(conf: DatasetConfig,
                         comm: CommunicatorWrapper = None,
                         rng=None,
                         resource_dir=DEFAULT_RESOURCE_DIR):
    # todo: use image-classification/imagenet utils

    if not os.path.exists(conf.dataset_root_dir):
        raise ValueError(f"[ImagenetDataIterator] '{conf.dataset_root_dir}' is not found. "
                         "Please make sure that you specify the correct directory path.")

    # extract label id
    label_wordnetid_csv = os.path.join(resource_dir, "label_wordnetid.csv")
    _info(f"load label_wordnetid from {label_wordnetid_csv}.")
    dname2label = {}
    with open(label_wordnetid_csv, "r") as fp:
        for l in fp:
            label, dname = l.rstrip().split(",")
            dname2label[dname] = int(label)

    # get all files
    if conf.train:
        # ilsvrcYYYY/train/{label id}/*.JPEG
        root_dir = pathlib.Path(os.path.join(conf.dataset_root_dir, "train"))
        _info(f"load train data and label from {root_dir}.")

        raw_paths = sorted(root_dir.rglob('*.JPEG'))
        paths = []
        labels = []

        for path in raw_paths:
            # Have to change pathlib.Path to string to avoid imread error
            paths.append(str(path))

            # Extract label
            name = path.name
            dname = name.split("_")[0]
            label = dname2label[dname]
            labels.append(label)

    else:
        # ilsvrsYYYY/val/*.JPEG
        root_dir = os.path.join(conf.dataset_root_dir, "val")
        _info(f"load validation data from {root_dir}.")
        raise NotImplementedError("val is not supported now.")

    ds = SimpleDatasource(conf,
                          img_paths=paths,
                          labels=labels,
                          rng=rng)

    _info(f"Loaded imagenet dataset. # of images: {ds.size}.")

    ds = _get_sliced_data_source(ds, comm, conf.shuffle_dataset)

    return data_iterator(ds,
                         conf.batch_size,
                         with_memory_cache=False,
                         use_thread=True,
                         with_file_cache=False)


def test_data_iterator(di, output_dir, comm=None, num_iters=100):
    from neu.reporter import KVReporter
    from nnabla.utils.image_utils import imsave

    reporter = KVReporter(comm=comm)

    os.makedirs(output_dir, exist_ok=True)

    for itr in range(num_iters):
        data, label = di.next()

        reporter.kv_mean("mean", data.mean())
        reporter.kv_mean("std", data.std())
        reporter.kv_mean("max", data.max())
        reporter.kv_mean("min", data.min())

        imsave(os.path.join(
            output_dir, f"{itr}.png"), data, channel_first=True)

    reporter.dump()


@click.command()
@click.option("--imagenet_base_dir", default=None)
def main(imagenet_base_dir):
    from neu.misc import init_nnabla
    comm = init_nnabla(ext_name="cpu", device_id=0, type_config="float")

    if imagenet_base_dir is not None and os.path.exists(imagenet_base_dir):
        logger.info("Test imagenet data iterator.")
        di = ImagenetDataIterator(2, imagenet_base_dir, comm=comm)
        test_data_iterator(di, "./tmp/imagene", comm)


if __name__ == "__main__":
    main()


__all__ = [ImagenetDataIterator]
