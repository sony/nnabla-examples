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

from neu.tts.hparams import HParams

hparams = HParams(

    # dataset parameters
    corpus_path='/speech/misc/denguyeb1/data/ljspeech/LJSpeech-1.1/',
    precomputed_path='/home/denguyeb/Desktop/datasets/hifi/',

    output_path="./log/hifigan/",      # directory to save results

    segment_length=8192,
    resblock_kernel_sizes=[3, 7, 11],
    resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    upsample_initial_channel=512,
    resblock="1",
    upsample_rates=[8, 8, 2, 2],
    periods=[2, 3, 5, 7, 11],
    n_layers_D=5,                       # number of layers in discriminators

    # spectrogram parameters
    sr=22050,                           # sampling rate used to read audios
    # length of the windowed signal after padding with zeros.
    n_fft=1024,
    n_mels=80,                          # number of mel filters
    mel_fmin=0.0,                       # minimum mel bank
    mel_fmax=8000.0,                    # maximum mel bank
    hop_length=256,
    win_length=1024,                    # window length

    seed=123456,                        # random seed

    # optimization parameters
    batch_size=16,                      # batch size
    epoch=3501,                         # number of epochs
    print_frequency=25,                 # number of iterations before printing
                                        # to log file
    epochs_per_checkpoint=100,          # number of epochs for each checkpoint

    alpha=2e-4,                         # learning rate
    beta1=0.8,
    beta2=0.99,
    lr_decay=0.999,
)
