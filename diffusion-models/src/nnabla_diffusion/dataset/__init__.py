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

from neu.comm import CommunicatorWrapper
from nnabla_diffusion.config import DatasetConfig

from .cifar10 import Cifar10DataIterator
from .common import SimpleDataIterator
from .imagenet import ImagenetDataIterator
from .laion import Laion400mDataIterator

# avoid an error for loading trancated image with pillow
try:
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
except:
    pass

iterators = {
    "cifar10": Cifar10DataIterator,
    "imagenet": ImagenetDataIterator,
    "laion-400m": Laion400mDataIterator,
}


def get_dataset(conf: DatasetConfig, comm: CommunicatorWrapper):
    data_iterator = iterators.get(conf.name, SimpleDataIterator)
    return data_iterator(conf, comm=comm)
