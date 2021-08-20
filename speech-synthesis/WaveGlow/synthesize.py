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
sys.path.append(str(Path().cwd().parents[1] / 'utils'))

import nnabla as nn
from nnabla.ext_utils import get_extension_context
import numpy as np
from scipy.io import wavfile

from hparams import hparams as hp
from model.denoiser import Denoiser
from model.model import WaveGlow


def synthesize(args):
    hp.batch_size = 1
    # load the model
    model = WaveGlow(hp)
    model.training = False
    model.load_parameters(args.f_model, raise_if_missing=True)

    # build a denoiser
    denoise = Denoiser(model, hp)

    x_mel = nn.Variable.from_numpy_array(np.load(args.f_mel))
    o_aud = model.infer(x_mel)

    o_aud.forward(clear_buffer=True)
    wave = denoise(o_aud.d[0].copy(), strength=args.noise_level)
    wavfile.write(args.f_output, rate=hp.sr, data=wave)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--context', '-c', type=str, default='cudnn',
                        help="Extension module. 'cudnn' is highly recommended.")
    parser.add_argument("--device-id", "-d", type=str, default='-1',
                        help='A list of device ids to use.\
                        This is only valid if you specify `-c cudnn`. \
                        Defaults to use all available GPUs.')
    parser.add_argument("--noise-level", "-l", type=float, default=0.003,
                        help='Noise level.')
    parser.add_argument("--f-model", "-m", type=str,
                        help='File path to the trained model.')
    parser.add_argument("--f-mel", "-f", type=str,
                        help='File path to the mel file.')
    parser.add_argument("--f-output", "-o", type=str, default='sample.wav',
                        help='File path to the synthetic output waveform.')
    args = parser.parse_args()

    # setup context for nnabla
    if args.device_id != '-1':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    # setup nnabla context
    ctx = get_extension_context(args.context, device_id='0')
    nn.set_default_context(ctx)

    synthesize(args)
