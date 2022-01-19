# Copyright 2022 Sony Group Corporation.
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


def get_args():
    """
    parse args
    """
    description = "Dataset Debiasing by GAN"
    parser = argparse.ArgumentParser(description)
    parser.add_argument("-d", "--device-id", type=int, default=0,
                        help="Device ID of the GPU for training")
    parser.add_argument("-c", "--context", type=str, default="cudnn",
                        help="Extension path: cpu or cudnn.")
    parser.add_argument("--type-config", "-t", type=str, default='float',
                        help='Type of computation. e.g. "float", "half".')

    parser.add_argument('--model_train', choices=['baseline', 'gan_debiased'],
                        type=str, default='baseline',
                        help="baseline model or gan_debiased model")
    parser.add_argument("--base-img-path", type=str,
                        default="data/celeba", help="original image path.")
    parser.add_argument('--attribute', type=int, default=1,
                        help="choose the target attribute to train "
                             "(e.g., Arched EyeBrows (1), Bushy Eyebrows(12), smilling (31),etc..")
    parser.add_argument('--protected_attribute', type=int, default=20,
                        help="choose the protected attributes (e.g., Male (20), Pale_Skin (26))")
    # as per author's citation, the baseline model is trained on the CelebA training set
    # with 162,770 images.
    parser.add_argument("--train-beg", type=int, default=0,
                        help="start the training samples")
    parser.add_argument("--valid-beg", type=int, default=162770,
                        help="start the validation sample")
    parser.add_argument("--test-beg", type=int, default=182610,
                        help="start the test sample")
    parser.add_argument("--batch-size", "-b", type=int, default=32,
                        help="Batch size.")
    parser.add_argument("--learning-rate", "-l", type=float, default=1e-4,
                        help="learning rate is a configurable hyper parameter "
                             "used in the training of neural network")
    parser.add_argument('--max_iter_base', type=int, default=5086,
                        help='max iteration to train the baseline model for an epoch')
    parser.add_argument('--max_iter_gan_debiased', type=int, default=15086,
                        help='max iteration to train the gan-debiased model for an epoch')
    parser.add_argument("--model-save-path", "-o",
                        type=str, default='./results',
                        help='Path where the model parameters saved.')
    # gan-based settings
    parser.add_argument('--num_images', type=int, default=175000,
                        help="Number of images to be generated.")
    parser.add_argument("--latent", type=int, default=512,
                        help="Number of latent variables.")
    parser.add_argument('--generate',
                        choices=[
                            'orig',
                            'flip',
                        ], default='orig', help="generate original or flipped images")

    parser.add_argument('--generator_model', type=str,
                        default=r'./result/example_0/Gen_phase_128_epoch_4.h5',
                        help="Model load path used in generation and validation")

    parser.add_argument('--fake_data_dir', type=str, default='data/fake_images',
                        help='Save the generated images by the PGGAN'
                             '(original images :fake_data_dir/AllGenImages'
                             'flipped images :fake_data_dir/"name of the attribute"')
    parser.add_argument('--record_latent_vector', type=str, default='./results/GAN_model',
                        help='Save the latent vectors')

    opt = vars(parser.parse_args())
    return opt
