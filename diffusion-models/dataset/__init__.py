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

from .cifar10 import Cifar10DataIterator
from .imagenet import ImagenetDataIterator
from .common import SimpleDataIterator

from config import DatasetConfig
from neu.comm import CommunicatorWrapper

# avoid an error for loading trancated image with pillow
try:
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
except:
    pass


def get_dataset(conf: DatasetConfig, comm: CommunicatorWrapper):
    if conf.name == "cifar10":
        data_iterator = Cifar10DataIterator(conf, 
                                            comm=comm,
                                            train=True)
    elif conf.name == "imagenet":
        data_iterator = ImagenetDataIterator(conf,
                                             comm=comm,
                                             train=True)
    else:
        data_iterator = SimpleDataIterator(conf,
                                           comm=comm)

    return data_iterator
