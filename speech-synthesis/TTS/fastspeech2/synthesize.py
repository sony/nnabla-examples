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
import sys
from pathlib import Path

sys.path.append(str(Path().cwd().parents[2] / 'utils'))

import nnabla as nn
import numpy as np
from nnabla.ext_utils import get_extension_context

from hparams import hparams as hp
from model.model import FastSpeech2
from utils import text

from utils.text.wdict import ___VALID_SYMBOLS___


def synthesize(args):
    phone_data = text.text_to_phonemes(args.text)
    phone_id = text.phonemes_to_ids(phone_data)
    ph_len = len(phone_id)

    hp.max_len_phone = ph_len + 1
    hp.max_len_mel = 2000
    hp.batch_size = 1

    model = FastSpeech2(hp)

    model.training = False
    model.load_parameters(args.model, raise_if_missing=True)

    phone_inp = nn.Variable((hp.batch_size, hp.max_len_phone))
    phone_len = nn.Variable((hp.batch_size, 1))
    out = model(
        phone_inp, phone_len,
        control_pitch=args.pitch,
        control_energy=args.energy,
        control_duration=args.duration,
    )[1]
    phone_id = np.pad(phone_id, [0, hp.max_len_phone - ph_len],
                      constant_values=___VALID_SYMBOLS___.index('EOL'))

    phone_inp.d = phone_id.reshape(phone_inp.shape)
    phone_len.d = ph_len

    out.forward(clear_buffer=True)
    np.save(args.output, out.d[0].T)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--context', '-c', type=str, default='cudnn',
        help="Extension module. 'cudnn' is highly recommended.")
    parser.add_argument("--device-id", "-d", type=str, default='-1',
                        help='A list of device ids to use.\
                        This is only valid if you specify `-c cudnn`.\
                        Defaults to use all available GPUs.')
    parser.add_argument("--text", "-t", type=str, default="Hello world",
                        help="Input text")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="Path to the pre-trained model")
    parser.add_argument("--pitch", "-p", type=float, default=1.0,
                        help="Pitch control factor")
    parser.add_argument("--energy", "-e", type=float, default=1.0,
                        help="Energy control factor")
    parser.add_argument("--duration", "-u", type=float, default=1.0,
                        help="Duration control factor")
    parser.add_argument("--output", "-o", type=str, default='./out.npy',
                        help="Output wave file")

    args = parser.parse_args()
    # setup context for nnabla
    if args.device_id != '-1':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    ctx = get_extension_context(args.context, device_id='0')
    nn.set_default_context(ctx)

    synthesize(args)
