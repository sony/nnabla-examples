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

from genericpath import exists
import os
import sys
import argparse

stylegan2_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(stylegan2_path)
from stylegan2.generate import synthesis
from stylegan2.networks import mapping_network
from stylegan2.ops import lerp, convert_images_to_uint8

import numpy as np
from tqdm import trange

import nnabla as nn
from nnabla.ext_utils import get_extension_context
import nnabla.functions as F

import functools
from celeba_classifier import resnet_prediction


def generate_attribute_direction(args, attribute_prediction_model):

    if not os.path.isfile(os.path.join(args.weights_path, 'gen_params.h5')):
        os.makedirs(args.weights_path, exist_ok=True)
        print("Downloading the pretrained tf-converted weights. Please wait...")
        url = "https://nnabla.org/pretrained-models/nnabla-examples/GANs/stylegan2/styleGAN2_G_params.h5"
        from nnabla.utils.data_source_loader import download
        download(url, os.path.join(args.weights_path, 'gen_params.h5'), False)

    nn.load_parameters(os.path.join(args.weights_path, 'gen_params.h5'))
    print('Loaded pretrained weights from tensorflow!')

    nn.load_parameters(args.classifier_weight_path)
    print(f'Loaded {args.classifier_weight_path}')

    batches = [args.batch_size for _ in range(
        args.num_images//args.batch_size)]
    if args.num_images % args.batch_size != 0:
        batches.append(args.num_images - (args.num_images //
                                          args.batch_size)*args.batch_size)

    w_plus, w_minus = 0.0, 0.0
    w_plus_count, w_minus_count = 0.0, 0.0
    pbar = trange(len(batches))
    for i in pbar:
        batch_size = batches[i]
        z = [F.randn(shape=(batch_size, 512)).data]

        z = [z[0], z[0]]

        for i in range(len(z)):
            z[i] = F.div2(z[i], F.pow_scalar(F.add_scalar(F.mean(
                z[i] ** 2., axis=1, keepdims=True), 1e-8), 0.5, inplace=True))

        # get latent code
        w = [mapping_network(z[0], outmaps=512, num_layers=8)]
        w += [mapping_network(z[1], outmaps=512, num_layers=8)]

        # truncation trick
        dlatent_avg = nn.parameter.get_parameter_or_create(
            name="dlatent_avg", shape=(1, 512))
        w = [lerp(dlatent_avg, _, 0.7) for _ in w]

        constant_bc = nn.parameter.get_parameter_or_create(
                        name="G_synthesis/4x4/Const/const",
                        shape=(1, 512, 4, 4))
        constant_bc = F.broadcast(
            constant_bc, (batch_size,) + constant_bc.shape[1:])

        gen = synthesis(w, constant_bc, noise_seed=100, mix_after=7)

        classifier_score = F.softmax(attribute_prediction_model(gen, True))
        confidence, class_pred = F.max(
            classifier_score, axis=1, with_index=True, keepdims=True)

        w_plus += np.sum(w[0].data*(class_pred.data == 0) *
                         (confidence.data > 0.65), axis=0, keepdims=True)
        w_minus += np.sum(w[0].data*(class_pred.data == 1)
                          * (confidence.data > 0.65), axis=0, keepdims=True)

        w_plus_count += np.sum((class_pred.data == 0)*(confidence.data > 0.65))
        w_minus_count += np.sum((class_pred.data == 1)
                                * (confidence.data > 0.65))

        pbar.set_description(f'{w_plus_count} {w_minus_count}')

    # save attribute direction
    attribute_variation_direction = (
        w_plus/w_plus_count) - (w_minus/w_minus_count)
    print(w_plus_count, w_minus_count)
    np.save(
        f'{args.classifier_weight_path.split("/")[0]}/direction.npy', attribute_variation_direction)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classifier-weight-path', type=str, default='bangs/params_001300.h5',
                        help="Path to pretrained classifier parameters")
    parser.add_argument('--weights-path', type=str, default='./',
                        help="Path to store pretrained stylegan2 parameters")

    parser.add_argument('--batch-size', type=int, default=4,
                        help="Batch-size of 1 forward pass of the generator")
    parser.add_argument('--num-images', type=int, default=50000,
                        help="Number of images to use to generate the direcion.")

    parser.add_argument('--context', type=str, default="cudnn",
                        help="context. cudnn is recommended.")

    args = parser.parse_args()

    assert args.num_images > args.batch_size, 'Number of images must be more than the batch-size'

    ctx = get_extension_context(args.context)
    nn.set_default_context(ctx)
    nn.set_auto_forward(True)

    attribute_prediction_model = functools.partial(
        resnet_prediction, nmaps=64, act=F.relu)

    generate_attribute_direction(args, attribute_prediction_model)


if __name__ == '__main__':
    main()
