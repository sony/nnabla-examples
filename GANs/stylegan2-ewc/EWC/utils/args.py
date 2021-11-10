# Copyright 2021 Sony Corporation.
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

from argparse import ArgumentParser


def make_parser():
    parser = ArgumentParser(
        description='Few-shot image generation with EWC: Nnabla implementation')

    parser.add_argument('--config', type=str, default='configs/ffhq.yaml',
                        help='Path to config file')

    # # [few-shot learning]
    parser.add_argument('--pre_trained_model', type=str, default='path to pre trained model',
                        help='Path to trained model weights')
    parser.add_argument('--ewc_weight_path', type=str, default='../result/FisherInformation.npz',
                        help='Path to the calcucated fisher information result.')
    parser.add_argument('--ewc_iter', type=int, default=100,
                        help='The number of iterations for the calculation of fisher information')

    parser.add_argument('--extension_module', type=str, default='cudnn',
                        help='Device context')
    parser.add_argument('--device_id', type=str, default='0',
                        help='Device Id')

    parser.add_argument('--img_size', type=int, default=256,
                        help='Image size to generate')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Image size to generate')

    return parser
