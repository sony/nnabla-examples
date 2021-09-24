# Copyright (c) 2017 Sony Corporation. All Rights Reserved.
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

import argparse
import os
from pathlib import Path

import sys
sys.path.append(str(Path().cwd().parents[1] / 'utils'))

import nnabla as nn
import numpy as np
from neu.comm import CommunicatorWrapper
from neu.tts.optimizer import Optimizer
from nnabla.ext_utils import get_extension_context
from nnabla.logger import logger
from nnabla.utils.data_iterator import data_iterator
from nnabla.utils.learning_rate_scheduler import ExponentialScheduler

from dataset import LJSpeechDataSource
from hparams import hparams as hp
from model.model import Discriminator, Generator
from train import HiFiGANTrainer


def run(args):
    """Runs the algorithm."""
    Path(hp.output_path).mkdir(parents=True, exist_ok=True)

    # setup nnabla context
    ctx = get_extension_context(args.context)
    nn.set_default_context(ctx)
    hp.comm = CommunicatorWrapper(ctx)

    if hp.comm.n_procs > 1 and hp.comm.rank == 0:
        n_procs = hp.comm.n_procs
        logger.info(f'Distributed training with {n_procs} processes.')

    rng = np.random.RandomState(hp.seed)

    # train data
    train_loader = data_iterator(
        LJSpeechDataSource('meta_train.csv', hp, shuffle=True, rng=rng),
        batch_size=hp.batch_size, with_memory_cache=False
    )
    # valid data
    valid_loader = data_iterator(
        LJSpeechDataSource('meta_test.csv', hp, shuffle=False, rng=rng),
        batch_size=hp.batch_size, with_memory_cache=False
    )
    dataloader = dict(train=train_loader, valid=valid_loader)

    # build model
    gen = Generator(hp)
    dis = Discriminator(hp)

    # setup optimizer
    iter_interval = train_loader.size//hp.batch_size
    g_optim = Optimizer(
        lr_scheduler=ExponentialScheduler(
            hp.alpha, gamma=hp.lr_decay, iter_interval=iter_interval),
        name='AdamW', alpha=hp.alpha, beta1=hp.beta1, beta2=hp.beta2
    )
    d_optim = Optimizer(
        lr_scheduler=ExponentialScheduler(
            hp.alpha, gamma=hp.lr_decay, iter_interval=iter_interval),
        name='AdamW', alpha=hp.alpha, beta1=hp.beta1, beta2=hp.beta2
    )

    HiFiGANTrainer(gen, dis, g_optim, d_optim, dataloader, hp).run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--context', '-c', type=str, default='cudnn',
                        help="'cudnn' is highly recommended.")
    parser.add_argument("--device-id", "-d", type=str, default='-1',
                        help='A list of device ids to use.\
                        This is only valid if you specify `-c cudnn`. \
                        Defaults to use all available GPUs.')
    for key, value in hp.__dict__.items():
        name = "--" + key
        if type(value) == list:
            nargs, t = '+', type(value[0])
        else:
            nargs, t = None, type(value)
        parser.add_argument(name, type=t, nargs=nargs, default=value)

    args = parser.parse_args()
    for k, v in vars(args).items():
        hp.__dict__[k] = v

    # setup context for nnabla
    if args.device_id != '-1':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    run(args)
