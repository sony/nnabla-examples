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

import argparse
import os
from pathlib import Path

import sys
sys.path.append(str(Path().cwd().parents[2] / 'utils'))
from neu.comm import CommunicatorWrapper
from neu.tts.optimizer import Optimizer

import nnabla as nn
from nnabla.ext_utils import get_extension_context
from nnabla.logger import logger
from nnabla.utils.data_iterator import data_iterator
from nnabla_ext.cuda import StreamEventHandler
import numpy as np

from dataset import LJSpeechDataSource
from hparams import hparams as hp
from model.model import Tacotron
from train import TacotronTrainer
from scheduler import NoamScheduler


def run(args):
    """Runs the algorithm."""
    Path(hp.output_path).mkdir(parents=True, exist_ok=True)

    # setup nnabla context
    ctx = get_extension_context(args.context, device_id='0')
    nn.set_default_context(ctx)
    hp.comm = CommunicatorWrapper(ctx)
    hp.event = StreamEventHandler(int(hp.comm.ctx.device_id))

    if hp.comm.n_procs > 1 and hp.comm.rank == 0:
        n_procs = hp.comm.n_procs
        logger.info(f'Distributed training with {n_procs} processes.')

    rng = np.random.RandomState(hp.seed)

    # setup optimizer
    lr_scheduler = NoamScheduler(hp.alpha, warmup=hp.warmup)
    optimizer = Optimizer(
        weight_decay=hp.weight_decay,
        max_norm=hp.max_norm,
        lr_scheduler=lr_scheduler,
        name='Adam', alpha=hp.alpha
    )

    # train data
    train_loader = data_iterator(
        LJSpeechDataSource('metadata_train.csv', hp, shuffle=True, rng=rng),
        batch_size=hp.batch_size, with_memory_cache=False
    )
    # valid data
    valid_loader = data_iterator(
        LJSpeechDataSource('metadata_valid.csv', hp, shuffle=False, rng=rng),
        batch_size=hp.batch_size, with_memory_cache=False
    )
    dataloader = dict(train=train_loader, valid=valid_loader)
    model = Tacotron(hp)

    TacotronTrainer(model, dataloader, optimizer, hp).run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--context', '-c', type=str, default='cudnn',
                        help="Extension module. 'cudnn' is highly recommended.")
    parser.add_argument("--device-id", "-d", type=str, default='-1',
                        help='A list of device ids to use.\
                        This is only valid if you specify `-c cudnn`. \
                        Defaults to use all available GPUs.')
    args = parser.parse_args()

    # setup context for nnabla
    if args.device_id != '-1':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    run(args)
