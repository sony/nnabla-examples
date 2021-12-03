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

# avoid an error for loading trancated image with pillow
try:
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
except:
    pass


def get_dataset(args, comm):
    assert hasattr(args, "dataset")
    assert hasattr(args, "image_shape")
    assert len(args.image_shape) == 4
    assert hasattr(args, "data_dir")

    if args.dataset == "cifar10":
        data_iterator = Cifar10DataIterator(
            args.batch_size, comm=comm, train=True)
    elif args.dataset.startswith("imagenet") and max(*args.image_shape[-2:]) > 64:
        data_iterator = ImagenetDataIterator(args.batch_size,
                                             args.dataset_root_dir,
                                             image_size=args.image_shape[-2:],
                                             fix_aspect_ratio=args.fix_aspect_ratio,
                                             comm=comm,
                                             train=True)
    else:
        data_iterator = SimpleDataIterator(args.batch_size,
                                           args.dataset_root_dir,
                                           image_size=args.image_shape[-2:],
                                           comm=comm, on_memory=args.dataset_on_memory,
                                           fix_aspect_ratio=args.fix_aspect_ratio)

    return data_iterator
