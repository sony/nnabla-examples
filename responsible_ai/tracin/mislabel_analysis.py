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
import cv2
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from distutils.util import strtobool
from glob import glob


def accumulate_rates(score, raw, shuffle):
    check_rate = [round(i, 2) for i in np.arange(0, 1.1, 0.01)]
    fraction_rate, data_rate = np.array([]), np.array([])

    indices = np.argsort(-score)
    all_num = len(raw)
    fraction_num = len(np.where(raw != shuffle)[0])

    for threshold in check_rate:
        checked_fraction = 0
        check_num = int(all_num * threshold)
        check_data = indices[:check_num]
        for ind in check_data:
            if raw[ind] != shuffle[ind]:
                checked_fraction += 1
        data_rate = np.append(data_rate, threshold)
        fraction_rate = np.append(fraction_rate,
                                  float(checked_fraction / fraction_num))
    return fraction_rate, data_rate


def save_score_curve(score, raw, shuffle):

    plt.figure()
    fraction_rate, data_rate = accumulate_rates(score, raw, shuffle)
    plt.plot(data_rate, fraction_rate)
    plt.xlabel('Fraction of training data checked')
    plt.ylabel('Fraction of mislabeled identified')
    plt.grid(color='gray', linewidth=1, alpha=0.5)
    plt.savefig(os.path.join(args.output, 'score_curve.png'))
    plt.close()


def save_distribution(score, raw, shuffle):
    correct = score[np.where(raw == shuffle)[0]]
    wrong = score[np.where(raw != shuffle)[0]]
    plt.figure()

    sns.distplot(correct, label='correct_label', color='blue')
    sns.distplot(wrong, label='shuffled_label', color='red')
    plt.legend()
    plt.xlabel('Self Influence')
    plt.savefig(os.path.join(args.output, 'self_influence_distribution.png'))
    plt.close()


def data_load(input_dir):
    images = np.load(os.path.join(input_dir, 'x_train.npy'))
    influence = np.load(os.path.join(input_dir, 'influence_all_epoch.npy'))
    label_raw = np.load(os.path.join(input_dir, 'y_train.npy'))
    label_shuffle = np.load(os.path.join(input_dir, 'y_shuffle_train.npy'))

    return images, influence, label_raw, label_shuffle


def rank_ckpt_influence(raw, shuffle, extract_percentage=0.10):
    '''
    analysis of the trend of influence samples between extracted checkpoints
    '''
    raw, shuffle = np.squeeze(raw), np.squeeze(shuffle)
    labels = ['airplane', 'automobile', 'bird', 'cat',
              'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    ckpt_influences = sorted(glob(os.path.join(args.input, '*_influence.npy')))
    for ckpt_influence in ckpt_influences:
        counter = np.zeros(10)
        influence = np.load(ckpt_influence)
        extract_num = int(len(influence) * extract_percentage)
        indices = np.argsort(-influence)[:extract_num]
        for indice in indices:
            if raw[indice] != shuffle[indice]:
                counter[raw[indice]] += 1
        epoch = os.path.basename(ckpt_influence).split('_')[0]
        plt.figure(figsize=(8, 5))
        plt.barh(np.arange(len(labels)), counter)
        plt.yticks(np.arange(len(labels)), labels)
        plt.title(f'Epoch:{epoch} top_influence_sample')
        plt.xlabel('Number')
        plt.ylabel('Label')
        plt.tight_layout()
        plt.savefig(os.path.join(
            args.output, f'{epoch}_top_influence_sample.png'))
        plt.close()


def save_extracted_images(images, score, raw, shuffle):
    labels_name = ['airplane', 'automobile', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    iteration = int(len(images)/10)
    check_num = [int(i) for i in np.arange(0, len(images), len(images)/10)]
    for num in check_num:
        # prepare save directory
        correct_dir = os.path.join(
            args.output, 'extracted_samples', str(num+iteration), 'correct_label')
        incorrect_dir = os.path.join(
            args.output, 'extracted_samples', str(num+iteration), 'incorrect_label')
        for dir in [correct_dir, incorrect_dir]:
            if not os.path.exists(dir):
                os.makedirs(dir)

        # save correct / incorrect samples
        indices = np.argsort(-score)[num:num+iteration]
        for ind in indices:
            if raw[ind] != shuffle[ind]:
                cv2.imwrite(os.path.join(incorrect_dir, f'{labels_name[int(raw[ind])]}_({labels_name[int(shuffle[ind])]}).jpg'), np.transpose(
                    images[ind], (1, 2, 0)))
            else:
                cv2.imwrite(os.path.join(correct_dir, f'{labels_name[int(raw[ind])]}_({labels_name[int(shuffle[ind])]}) .jpg'), np.transpose(
                    images[ind], (1, 2, 0)))


def main():
    print('save various figures ...')
    images, score, raw, shuffle = data_load(args.input)
    save_distribution(score, raw, shuffle)
    save_score_curve(score, raw, shuffle)
    if args.ckpt_analysis:
        print('analyse checkpoint...')
        rank_ckpt_influence(raw, shuffle)
    if args.save_extracted:
        print('saving extracted samples ...')
        save_extracted_images(images, score, raw, shuffle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='evaluation classification')
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--ckpt_analysis', type=strtobool, default=False,
                        help='whether verify the difference of most influence sample trend between checkpoint or not')
    parser.add_argument('--save_extracted', type=strtobool, default=False,
                        help='whethre save the extracted images at certain threshold of self influence score')
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    main()
