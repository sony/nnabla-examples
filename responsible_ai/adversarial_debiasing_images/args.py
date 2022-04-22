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
    description = "Adversarial debiasing"
    parser = argparse.ArgumentParser(description)
    parser.add_argument("-d", "--device-id", type=int, default=0,
                        help="Device ID of the GPU for training")
    parser.add_argument("-c", "--context", type=str, default="cudnn",
                        help="Extension path: cpu or cudnn.")
    parser.add_argument("--type-config", "-t", type=str, default='float',
                        help='Type of computation. e.g. "float", "half".')

    parser.add_argument('--model_train', choices=['baseline', 'adversarial'],
                        type=str, default='adversarial',
                        help="baseline model or Adversarial de-biasing ")
    parser.add_argument("--celeba_image_train_dir", type=str,
                        default="./data/train", help="Path of training directory")
    parser.add_argument("--celeba_image_valid_dir", type=str,
                        default="./data/valid", help="Path of validation directory")
    parser.add_argument("--attr_path", type=str, default=r'./data/celeba/list_attr_celeba.txt',
                        help="celebA attribute file path (ex: list_attr_celeba.txt)")

    parser.add_argument('--attribute', type=str, default='Attractive',
                        help="choose the target attribute to train "
                             "(e.g., Attractive, Arched EyeBrows, Bushy Eyebrows, smilling ,etc..")
    parser.add_argument('--protected_attribute', type=str, default='Male',
                        help="choose the protected attributes (e.g., Male , Pale_Skin)")

    # as per author's citation, the baseline model is trained on the CelebA training set
    # with 162,770 images.
    parser.add_argument("--train-beg", type=int, default=0,
                        help="start the training samples")
    parser.add_argument("--valid-beg", type=int, default=162752,
                        help="start the validation sample")
    parser.add_argument("--test-beg", type=int, default=182592,
                        help="start the test sample")
    parser.add_argument("--batch-size", "-b", type=int, default=32,
                        help="Batch size.")
    parser.add_argument("--learning-rate", "-l", type=float, default=1e-4,
                        help="learning rate is a configurable hyper parameter "
                             "used in the training of neural network")
    parser.add_argument('--max_iter', type=int, default=5086,
                        help='max iteration to train the model for an epoch')
    parser.add_argument('--total_epochs', type=int, default=50,
                        help='total no of epochs to train the model')
    # Adversary traning parametrs
    parser.add_argument('--training_ratio', type=int, default=3,
                        help='training ratio b/w the classifier and adversary n/w')
    parser.add_argument('--lamda', type=int, default=5,
                        help='lambda parameter weighs the adversarial loss of each class')

    # save model
    parser.add_argument("--model-save-path", "-o",
                        type=str, default='./results',
                        help='path where the model parameters saved.')
    parser.add_argument("--model-load-path",
                        type=str, default=None,
                        help='path where to load the .nnp model')

    opt = vars(parser.parse_args())
    return opt
