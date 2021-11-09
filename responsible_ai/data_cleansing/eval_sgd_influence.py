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

import os
import argparse
import shutil
import pandas as pd
import nnabla as nn
from sgd_influence.utils import ensure_dir, get_context
from sgd_influence.eval import run_train_for_eval
from sgd_influence.model import get_config
from sgd_influence.train import train
from sgd_influence.infl import infl_sgd
from sgd_influence.analyze import analyze
from datasets.create_cifar10_csv import create_data_csv as create_cifar10_csv
from datasets.create_stl10_csv import create_data_csv as create_stl10_csv
from datasets.create_mnist_csv import create_data_csv as create_mnist_csv


method_dict = {
    # sgd_Hara_retrain_last
    'Hara':
        {
            'calc_infl_method': 'last',
            'calc_infl_with_all_params': True,
            'shuffle_infl': False,
        },
    # sgd_Modified_retrain_last
    'Ours':
        {
            'calc_infl_method': 'last',
            'calc_infl_with_all_params': False,
            'shuffle_infl': False,
        },
    # random drop
    'Random':
        {
            'calc_infl_method': 'last',
            'calc_infl_with_all_params': False,
            'shuffle_infl': True,
        },
}


def has_enough_data(data_csv_dir, num_of_data):
    _df = pd.read_csv(data_csv_dir)
    if num_of_data >= len(_df):
        print(
            f'n_to_remove ({num_of_data}) >= total dataset size ({len(_df)})')
        return False
    return True


def make_random_file(infl_file):
    infl = pd.read_csv(infl_file)
    infl = infl.sample(frac=1)
    infl.to_csv(infl_file, index=False)


def train_and_infl(args, shuffle_infl):
    args.score_output = None
    config = get_config(args)
    model_info_dict = config.model_info_dict
    file_dir_dict = config.file_dir_dict
    need_evaluate = config.need_evaluate
    calc_infl_with_all_params = args.calc_infl_with_all_params
    print(file_dir_dict['infl_filename'])
    if os.path.exists(file_dir_dict['infl_filename']):
        return
    # train
    # only for the round where all params are calculated to avoid duplicated calculation
    if calc_infl_with_all_params:
        train(model_info_dict, file_dir_dict,
              calc_infl_with_all_params, need_evaluate)
    # infl
    if shuffle_infl:
        from_path = os.path.join(
            os.path.dirname(os.path.dirname(args.output)),
            next(iter(method_dict.keys()))
        )
        files = os.listdir(from_path)
        fn = [f for f in files if f'_{str(args.seed)}' in f][0]
        shutil.copy(os.path.join(from_path, fn), args.output)
        make_random_file(args.output)
    else:
        infl_sgd(model_info_dict, file_dir_dict,
                 calc_infl_with_all_params, need_evaluate)


def eval(args, method_score_dir, score_basename):
    for idx, n_to_remove in enumerate(args.remove_n_list):
        args.score_output = os.path.join(
            method_score_dir, score_basename + str(n_to_remove) + '_seed_' + str(args.seed) + '.csv')
        print(args.score_output)
        if os.path.exists(args.score_output):
            print('already exists')
            continue
        config = get_config(args, is_eval=True)
        model_info_dict = config.model_info_dict
        file_dir_dict = config.file_dir_dict
        if not has_enough_data(file_dir_dict['train_csv'], n_to_remove):
            continue
        run_train_for_eval(model_info_dict, file_dir_dict,
                           n_to_remove, args.retrain_all)


def create_csv_files(dataset_name, seed: int):
    func_dict = {
        'cifar10': create_cifar10_csv,
        'stl10': create_stl10_csv,
        'mnist': create_mnist_csv,
    }
    func_dict[dataset_name](seed)


def generate_dataset(dataset_name, seed):
    fd = {}
    base_dir = os.path.join(os.path.abspath(
        os.path.dirname(__file__)), 'datasets', dataset_name)
    fd['training'] = os.path.join(
        base_dir, f'{dataset_name}_training_{str(seed)}.csv')
    fd['validation'] = os.path.join(
        base_dir, f'{dataset_name}_validation_{str(seed)}.csv')
    fd['test'] = os.path.join(base_dir, f'{dataset_name}_test.csv')
    if os.path.exists(fd['training']) & os.path.exists(fd['validation']):
        pass
    else:
        create_csv_files(dataset_name, seed)
    return fd


def run(args):
    os.environ['NNABLA_CUDNN_DETERMINISTIC'] = '1'
    os.environ['NNABLA_CUDNN_ALGORITHM_BY_HEURISTIC'] = '1'
    # gpu/cpu
    ctx = get_context(args.device_id)
    nn.set_default_context(ctx)

    seeds = [i for i in range(args.n_trials)]
    weight_dir = 'infl_results_sgd'
    score_basename = 'score_delete_worst_'
    output_dir = args.output_dir

    for i, seed in enumerate(seeds):
        args.seed = seed
        fd = generate_dataset(args.dataset, seed)
        args.input_train, args.input_val, args.input_test = fd[
            'training'], fd['validation'], fd['test']
        for method_name, calc_setting in method_dict.items():
            # setup params
            method_score_dir = os.path.join(output_dir, 'score', method_name)
            method_infl_dir = os.path.join(
                output_dir, 'influence', method_name)
            ensure_dir(method_score_dir)
            ensure_dir(method_infl_dir)
            args.calc_infl_method = calc_setting['calc_infl_method']
            args.calc_infl_with_all_params = calc_setting['calc_infl_with_all_params']
            shuffle_infl = calc_setting['shuffle_infl']
            args.weight_output = os.path.join(weight_dir, 'seed_%02d' % (seed))
            ensure_dir(args.weight_output)
            args.output = os.path.join(
                method_infl_dir, 'influence_' + method_name.lower() + '_' + str(seed) + '.csv')
            # train and infl
            train_and_infl(args, shuffle_infl)
            # eval
            eval(args, method_score_dir, score_basename)

    # show result
    analyze(output_dir)


def main():
    parser = argparse.ArgumentParser(
        description='check performance of SGD-influence', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-o', '--output_dir', help='path to output dir', default='output_sgd_influence')
    parser.add_argument(
        '-e', '--n_epochs', help="epoch for SGD influence calculation(int) default=20", default=20, type=int)
    parser.add_argument(
        '-nt', '--n_trials', help='number or trials ', default=6, type=int)
    parser.add_argument(
        '-r', '--remove_n_list', help="list of n of samples to remove. ex: '-r 10 20' makes [10, 20]",
        type=int, nargs='+', default=[0, 1, 10, 40, 100, 200, 600, 1000, 3000, 5000, 10000, 30000])
    parser.add_argument(
        '-ds', '--dataset', choices=['cifar10', 'stl10', 'mnist'], default='cifar10')
    parser.add_argument(
        '-di', '--device-id', help='device id of gpu', default=0)
    parser.add_argument(
        '-ra', '--retrain_all', help='if True, retrain all.', action='store_true')
    parser.set_defaults(func=run)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
