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

from neu.tts.text import text_normalize
import nnabla as nn
from nnabla.ext_utils import get_extension_context
import numpy as np
from scipy.io import wavfile

from hparams import hparams as hp
from model.model import Tacotron2


def synthesize(args):
    char2idx = {ch: i for i, ch in enumerate(hp.vocab)}
    with open(args.f_text, 'r') as file:
        text = ''.join(file.readlines())
    # normalize the text
    text = text_normalize(text, hp.vocab)
    if len(text) >= hp.text_len - 1:
        text = text[:hp.text_len-1]
    text += '~'*(hp.text_len-len(text))
    text = np.array([char2idx[ch] for ch in text]).reshape(-1)

    hp.batch_size = 1
    # load the model
    model = Tacotron2(hp)
    model.training = False
    model.load_parameters(args.f_model)

    x_txt = nn.Variable([hp.batch_size, hp.text_len])
    _, mels, _, _ = model(x_txt)
    x_txt.d = text[np.newaxis, :]

    mels.forward(clear_buffer=True)
    m = mels.d.copy().reshape(1, -1, hp.n_mels)
    np.save(args.f_output, m.transpose((0, 2, 1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--context', '-c', type=str, default='cudnn',
                        help="Extension module. 'cudnn' is highly recommended.")
    parser.add_argument("--device-id", "-d", type=str, default=None,
                        help='A comma-separated list of device ids to use. \
                        This is only valid if you specify `-c cudnn`. \
                        Defaults to use all available GPUs.')
    parser.add_argument("--f-model", "-m", type=str,
                        help='File path to the trained model.')
    parser.add_argument("--f-text", "-f", type=str,
                        help='File path to the text file.')
    parser.add_argument("--f-output", "-o", type=str, default='sample.npy',
                        help='File path to the synthetic output waveform.')
    args = parser.parse_args()

    # setup context for nnabla
    if args.device_id is not None:
        try:
            device_list = ','.join(
                list(map(str, map(int, args.device_id.split(',')))))
        except ValueError:
            print(
                "--device-id requires a comma-separated list of GPU numbers", file=sys.stderr)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = device_list

    # setup nnabla context
    ctx = get_extension_context(args.context, device_id='0')
    nn.set_default_context(ctx)

    synthesize(args)
