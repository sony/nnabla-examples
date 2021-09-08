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

from neu.tts.hparams import HParams

hparams = HParams(

    # dataset parameters
    data_dir="./data/LJSpeech-1.1/",  # directory to the data
    # directory to save all precomputed FFTs
    save_data_dir="./data/LJSpeech-1.1/tacotron2",

    out_variables=["mel", "text", "gate"],  # which variables will be used

    mel_len=290,                        # frame length of mel spectrogram
    text_len=188,                       # maximum text length

    # spectrogram parameters
    sr=22050,                           # sampling rate used to read audios
    # length of windowed signal after padding with zeros.
    n_fft=1024,
    n_mels=80,                          # number of mel filters
    hop_length=256,                     # audio samples between adjacent STFT columns
    win_length=1024,                    # window length
    mel_fmin=0.0,                       # minimum mel bank
    mel_fmax=8000,                      # maximum mel bank

    # dictionary
    vocab="~ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!'(),-.:;?_ ",  # vocabulary
    # number of dimensions used for character embedding
    symbols_embedding_dim=512,

    # encoder parameters
    # number of dimensions used for encoder embedding
    encoder_embedding_dim=512,
    encoder_kernel_size=5,              # kernel size
    encoder_n_convolutions=3,           # number of convolutional layers

    # postnet parameters
    postnet_n_convolutions=5,           # number of convolutional layers
    postnet_kernel_size=5,              # kernel size
    postnet_embedding_dim=512,          # number of dimensions for decoder embedding

    # attention parameters
    attention_dim=128,                  # dimension of attention
    attention_location_kernel_size=31,  # kernel size
    attention_location_n_filters=32,    # number of filters
    attention_rnn_dim=1024,             # dimension of RNN
    p_attention_dropout=0.1,            # dropout rate

    # decoder parameters
    decoder_rnn_dim=1024,               # dimension of RNN for decoder
    p_decoder_dropout=0.1,              # dropout for decoder
    prenet_channels=[256, 256],         # number channels for prenet
    r=3,                                # number of frames generated on each timestep

    # optimization parameters
    batch_size=64,                      # batch size
    epoch=1501,                         # number of epochs
    print_frequency=50,                 # number of iterations before printing to log file
    output_path="./log/tacotron2/",     # directory to save results

    seed=123456,                        # random seed
    epochs_per_checkpoint=50,           # number of epochs for each checkpoint

    weight_decay=1e-6,                  # weight decay
    max_norm=1.0,                       # maximum norm used in clip_grad_by_norm
    alpha=1e-3,                         # learning rate
    warmup=4000,                        # number of iterations for warmup
    anneal_factor=0.1,                  # factor by which to anneal the learning rate
    # epoch at which to anneal the learning rate
    anneal_steps=(500, 1000, 1500)
)
