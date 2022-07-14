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
    description = "Reject Option-based Classification"
    parser = argparse.ArgumentParser(description)
    parser.add_argument("-d", "--device-id", type=int, default=0,
                        help="device ID of the GPU for training")
    parser.add_argument("-c", "--context", type=str, default="cudnn",
                        help="cpu or cudnn context for training")
    parser.add_argument("--type-config", "-t", type=str, default='float',
                        help='computation type, e.g. "float", "half"')
    parser.add_argument("--celeba_image_train_dir", type=str,
                        default="./data/train", help="path of training image directory")
    parser.add_argument("--celeba_image_valid_dir", type=str,
                        default="./data/valid", help="path of validation image directory")
    parser.add_argument("--celeba_image_test_dir", type=str,
                        default="./data/test", help="path of test image directory")
    parser.add_argument("--attr_path", type=str, default=r'./data/celeba/list_attr_celeba.txt',
                        help="celebA attribute file path and name "
                             "(ex: ./data/list_attr_celeba.txt)")
    parser.add_argument('--attribute', type=str, default='Attractive',
                        help="choose the target attribute to train "
                             "(e.g., Attractive, Arched EyeBrows, Bushy Eyebrows, smilling ,etc.")
    parser.add_argument('--protected_attribute', type=str, default='Male',
                        help="choose the protected attributes (e.g., Male , Pale_Skin)")
    parser.add_argument("--batch-size", "-b", type=int, default=32,
                        help="batch size")
    parser.add_argument("--learning-rate", "-l", type=float, default=1e-4,
                        help="learning rate is a critical hyper parameter "
                             "in most learning algorithms")
    parser.add_argument('--max_iter', type=int, default=5086,
                        help='max iterations per epoch')
    parser.add_argument('--total_epochs', type=int, default=20,
                        help='total no of epochs to train the model')
    # ROC
    parser.add_argument("--optimization_metric", type=str,
                        default="DPD", help="name of the metric to use for "
                                            "optimization - (Demographic parity difference (DPD), "
                                            "Absolute average odds difference (AAOD), "
                                            "Equal opportunity difference (EOD))")
    parser.add_argument("--metric_upper_bound", type=float,
                        default=0.10, help="upper bound constraint on the metric value")
    parser.add_argument("--metric_lower_bound", type=float,
                        default=0.0, help="lower bound constraint on the metric value")
    # save model
    parser.add_argument("--model-save-path", "-o",
                        type=str, default='./results',
                        help='path where the model parameters are saved')
    parser.add_argument("--model-load-path",
                        type=str, default=None, help='path where to load the .h5 model')

    opt = vars(parser.parse_args())

    return opt
