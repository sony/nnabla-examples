# Copyright 2020,2021 Sony Corporation.
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
import sys

import nnabla as nn
import numpy as np
from nnabla.logger import logger


def set_random_pseed(comm):
    x = nn.Variable.from_numpy_array(np.random.randint(low=0, high=1 << 30))
    comm.broadcast(x)

    from nnabla.random import set_parameter_seed
    set_parameter_seed(int(x.d))

    from nnabla.random import pseed
    logger.info(f"[rank {comm.rank}] seed: {pseed}")


def init_nnabla(conf=None, ext_name=None, device_id=None, type_config=None, random_pseed=True):
    import nnabla as nn
    from nnabla.ext_utils import get_extension_context

    from .comm import CommunicatorWrapper
    if conf is None:
        conf = AttrDict()
    if ext_name is not None:
        conf.ext_name = ext_name
    if device_id is not None:
        conf.device_id = device_id
    if type_config is not None:
        conf.type_config = type_config

    # set context
    ctx = get_extension_context(
        ext_name=conf.ext_name, device_id=conf.device_id, type_config=conf.type_config)

    # init communicator
    comm = CommunicatorWrapper(ctx)
    nn.set_default_context(comm.ctx)

    # set random seed for parameter
    if random_pseed:
        set_random_pseed(comm)

    # disable outputs from logger except rank==0
    if comm.rank > 0:
        import logging

        logger.setLevel(logging.ERROR)

    return comm


class AttrDict(dict):
    # special internal variable used for error message.
    _parent = []

    def __setattr__(self, key, value):
        if key == "_parent":
            self.__dict__["_parent"] = value
            return

        self[key] = value

    def __getattr__(self, key):
        if key not in self:
            raise AttributeError(
                "dict (AttrDict) has no chain of attributes '{}'".format(".".join(self._parent + [key])))

        if isinstance(self[key], dict):
            self[key] = AttrDict(self[key])
            self[key]._parent = self._parent + [key]

        return self[key]

    def dump_to_stdout(self):
        self.dump()

    def dump(self, file=sys.stdout, sort_keys=True):
        if not hasattr(file, "write"):
            assert isinstance(file, str)
            file = open(file, 'w')

        out = "\n================================configs================================\n"

        iterator = self.items()
        if sort_keys:
            iterator = sorted(iterator, key=lambda x: x[0])

        for k, v in iterator:
            out += "{}: {}\n".format(k, v)

        out += "======================================================================="

        print(out, file=file)


class DictInterfaceFactory(object):
    '''Creating a single dict interface of any function or class.

    Example:

    .. code-block:: python
        # Define a function.
        def foo(a, b=1, c=None):
            for k, v in locals():
                print(k, v)

        # Register the function to the factory.
        dictif = DictInterfaceFactory()
        dictif.register(foo)

        # You can call the registered function by name and a dict representing the arguments.
        cfg = dict(a=1, c='hello')
        dictif.call('foo', cfg)

        # The following will fail because the `foo` function requires `a`.
        #     cfg = dict(c='hello')
        #     dictif.call('foo', cfg)

        # Any argument not required will be just ignored.
        cfg = dict(a=1, aaa=0)
        dictif.call('foo', cfg)

        # You can also use it for class initializer (we use it as a class decorator).
        @dictif.register
        class Bar:
            def __init__(self, a, b, c=None):
                for k, v in locals():
                    print(k, v)

        bar = dictif.call('Bar', dict(a=0, b=0))

    '''

    def __init__(self):
        self._factory = {}

    def register(self, cls):
        import inspect

        # config interface function
        def func(cfg):
            sig = inspect.signature(cls)
            # Handle all arguments of the created class
            args = {}
            for p in sig.parameters.values():
                # Positional argument
                if p.default is p.empty and p.name not in cfg:
                    raise ValueError(
                        f'`{cls.__name__}`` requires an argument `{p.name}`. Not found in cfg={cfg}.')
                args[p.name] = cfg.get(p.name, p.default)
            return cls(**args)

        # Register config interface function
        self._factory[cls.__name__] = func
        return cls

    def call(self, name, cfg):
        if name in self._factory:
            return self._factory[name](cfg)
        raise ValueError(
            f'`{name}`` not found in `{list(self._factory.keys())}`.')


def makedirs(dirpath):
    if os.path.exists(dirpath):
        if os.path.isdir(dirpath):
            return
        else:
            raise ValueError(
                "{} already exists as a file not a directory.".format(dirpath))

    os.makedirs(dirpath)


def get_current_time():
    from datetime import datetime

    return datetime.now().strftime('%m%d_%H%M%S')


def get_iteration_per_epoch(dataset_size, batch_size, round="ceil"):
    """
    Calculate a number of iterations to see whole images in dataset (= 1 epoch).

    Args:
     dataset_size (int): A number of images in dataset
     batch_size (int): A number of batch_size.
     round (str): Round method. One of ["ceil", "floor"].

    return: int
    """
    import numpy as np

    round_func = {"ceil": np.ceil, "floor": np.floor}
    if round not in round_func:
        raise ValueError("Unknown rounding method {}. must be one of {}.".format(round,
                                                                                 list(round_func.keys())))

    ipe = float(dataset_size) / batch_size

    return int(round_func[round](ipe))
