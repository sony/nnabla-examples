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
import numpy as np
import csv

import nnabla as nn

import nnabla.functions as F
import nnabla.parametric_functions as PF

from calculate import KernelSHAP
from visualize import visualize
from train import train_model


def func(args):
    # Load csv

    with open(args.input, 'r') as f:
        reader = csv.reader(f)
        _ = next(reader)
        sample = np.array([[float(r) for r in row] for row in reader])[:, :-1]
    if len(sample.shape) == 1:
        sample = sample.reshape((1, sample.shape[0]))

    with open(args.train, 'r') as f:
        reader = csv.reader(f)
        feature_names = next(reader)[:-1]
        train = np.array([[float(r) for r in row] for row in reader])
    train, train_labels = train[:, :-1], (train[:, -1]).astype(int)
    train_labels = train_labels.reshape((len(train_labels), 1))

    # prepare variable
    data = nn.Variable(train.shape)
    X = nn.Variable(sample.shape)
    labels = nn.Variable(train_labels.shape)

    data.d = train
    X.d = sample
    labels.d = train_labels
    data.need_grad = True

    # define model
    def model(x):
        h = PF.affine(x, (32,), name='Affine')
        h = F.relu(h, True)
        h = PF.affine(h, (args.output_size,), name='Affine_2')
        h = F.softmax(h)
        return h

    # train model
    train_model(model, data, labels)

    # calculate and visualize SHAP
    Kernelshap = KernelSHAP(data, model, X, alpha=0)
    shap_values, expected_values = Kernelshap.calculate_shap()
    fig = visualize(expected_values, shap_values, sample,
                    feature_names, args.class_index, args.index)
    fig.savefig(args.output)


def main():
    parser = argparse.ArgumentParser(
        description='SHAP(tabular)\n'
                    '\n'
                    'A Unified Approach to Interpreting Model Predictions'
                    'Scott Lundberg, Su-In Lee'
                    'https://arxiv.org/abs/1705.07874\n'
                    '', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i', '--input', help='path to input csv file (csv)', required=True)
    parser.add_argument(
        '-t', '--train', help='path to training dataset csv file (csv)', required=True)
    parser.add_argument(
        '-os', '--output_size', help='output_size', required=True, type=int)
    parser.add_argument(
        '-i2', '--index', help='index to be explained (int), default=0', required=True, default=0, type=int)
    parser.add_argument(
        '-a', '--alpha', help='alpha of Ridge, default=0', required=True, default=0, type=float)
    parser.add_argument(
        '-c2', '--class_index', help='class index (int), default=0', required=True, default=0, type=int)
    parser.add_argument(
        '-o', '--output', help='path to output image, default=shap_tabular.png', required=True, default='shap_tabular.png')
    parser.set_defaults(func=func)
    args = parser.parse_args()

    args.func(args)


if __name__ == '__main__':
    main()
