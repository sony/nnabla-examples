import argparse
import os
import sys
from pathlib import Path

sys.path.append(str(Path().cwd().parents[2] / 'utils'))


import nnabla as nn
import numpy as np
from nnabla.ext_utils import get_extension_context
from nnabla.logger import logger
from nnabla.utils.data_iterator import data_iterator

from dataset import LJSpeechDataSource
from hparams import hparams as hp
from model.model import FastSpeech2
from train import Trainer
from neu.comm import CommunicatorWrapper
from neu.tts.optimizer import Optimizer
from utils.optim import NoamScheduler



def run(args):
    # create output path
    Path(hp.output_path).mkdir(parents=True, exist_ok=True)

    # setup nnabla context
    ctx = get_extension_context(args.context)
    nn.set_default_context(ctx)

    hp.comm = CommunicatorWrapper(ctx)
    if hp.comm.n_procs > 1 and hp.comm.rank == 0:
        n_procs = hp.comm.n_procs
        logger.info(f'Distributed training with {n_procs} processes.')

    rng = np.random.RandomState(hp.seed)

    train_loader = data_iterator(
        LJSpeechDataSource('meta_train.csv', hp, shuffle=True, rng=rng),
        batch_size=hp.batch_size, with_memory_cache=False, rng=rng
    )
    valid_loader = data_iterator(
        LJSpeechDataSource('meta_test.csv', hp, shuffle=False, rng=rng),
        batch_size=hp.batch_size, with_memory_cache=False, rng=rng
    )

    dataloader = dict(train=train_loader, valid=valid_loader)
    model = FastSpeech2(hp)
    optim = Optimizer(
        lr_scheduler=NoamScheduler(hp.lr, warmup=hp.warmup),
        name='Adam', alpha=hp.lr, beta1=hp.beta1, beta2=hp.beta2, eps=1e-9,
    )

    Trainer(model, optim, dataloader, rng, hp).run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--context', '-c', type=str, default='cudnn',
        help="Extension module. 'cudnn' is highly recommended.")
    parser.add_argument("--device-id", "-d", type=str, default='-1',
                        help='A list of device ids to use.\
                        This is only valid if you specify `-c cudnn`.\
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
