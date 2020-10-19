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

from utils.hparams import HParams

hparams = HParams(

    # dataset parameters
    dataset="LJSpeechDataSource",                  # which dataset to run
    data_dir="./data/LJSpeech-1.1/",               # directory to the data
    save_data_dir="./data/LJSpeech-1.1/tacotron",  # directory to save all precomputed FFTs
    out_variables=["mel", "linear", "text"],       # which variables will be used

    mel_len=162,                                   # maximum frame length of mel spectrogram
    text_len=188,                                  # maximum text length

    # spectrogram parameters
    sr=20000,                                      # sampling rate used to read audios
    n_fft=2048,                                    # length of windowed signal after padding with zeros.
    n_mels=80,                                     # number of mel filters
    hop_length=250,                                # audio samples between adjacent STFT columns
    win_length=1000,                               # window length
    ref_db=20,                                     # reference decibel
    max_db=100,                                    # maximum decibel
    mel_fmin=0.0,                                  # minimum mel bank
    mel_fmax=None,                                 # maximum mel bank
    preemphasis=0.97,                              # preemphasis factor

    # dictionary
    vocab="~ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!'(),-.:;?,_ ",

    r=5,                                           # number of frames generated on each timestep
    n_iter=60,                                     # number of iterations for Griffin-Lim
    power=1.5,                                     # power used for Griffin-Lim

    symbols_embedding_dim=256,                     # number of dimensions used for character embedding
    prenet_channels=(256, 128),                    # number channels for prenet
    encoder_embedding_dim=256,                     # number of dimensions used for encoder embedding
    attention_dim=256,                             # dimension of attention
    postnet_embedding_dim=256,                     # number of dimensions for decoder embedding

    batch_size=32,                                 # batch size
    epoch=1000,                                    # number of epochs
    print_frequency=50,                            # number of iterations before printing to log file
    weight_decay=0.0,                              # weight decay
    max_norm=1.0,                                  # maximum norm used in clip_grad_by_norm
    alpha=0.001,                                   # learning rate
    warmup=4000,                                   # number of iterations for warmup
    epochs_per_checkpoint=50,                      # number of epochs for each checkpoint
    output_path="./log/tacotron-32/",                 # directory to save results

    seed=123456,                                   # random seed
)
