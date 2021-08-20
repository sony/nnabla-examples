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

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from .utils import ensure_dir


def get_summary_df(file_dir, method_list, baseline_name):
    summary_df = pd.DataFrame()
    for method in method_list:
        file_method_dir = os.path.join(file_dir, method)
        score_files = os.listdir(file_method_dir)
        for score_file in score_files:
            df = pd.read_csv(os.path.join(file_method_dir, score_file))
            file_basename = score_file.split('_')
            seed = file_basename[-1].split('.')[0]
            n_drops = file_basename[-3]
            _df = df.copy()
            _df['method'] = method
            _df['seed'] = int(seed)
            _df['n_drops'] = int(n_drops)
            summary_df = pd.concat([summary_df, _df], axis=0)

    summary_df.loc[summary_df['n_drops'] == 0, 'method'] = baseline_name
    _v = summary_df.loc[summary_df['n_drops'] == 0].copy()
    for i in set(summary_df['n_drops']) - set([0]):
        _v['n_drops'] = i
        summary_df = pd.concat([summary_df, _v], axis=0)
    # calc mean
    mean_df = summary_df.groupby(['method', 'n_drops']).mean().reset_index()
    mean_df['mean'] = 'mean'
    mean_df['seed'] = np.nan
    # calc std
    std_df = summary_df.groupby(['method', 'n_drops']).std().reset_index()
    std_df['std'] = 'std'
    std_df['seed'] = np.nan
    summary_df = pd.concat([summary_df, mean_df], axis=0)
    summary_df = pd.concat([summary_df, std_df], axis=0)
    return summary_df.reset_index(drop=True)


def analyze_result(score_dir, baseline_name='No Removal'):
    # obtain method
    files = os.listdir(score_dir)
    methods = np.array(
        [f for f in files if os.path.isdir(os.path.join(score_dir, f))])
    # read data
    summary_df = get_summary_df(score_dir, methods, baseline_name)
    return summary_df


def plot_result(acc, output_dir, baseline_name, fig_save_dir_name='fig'):
    # drawing
    df = acc.loc[acc['mean'].notna()]
    hue = 'method'
    # sort labels
    labels = sorted(df[hue].drop_duplicates().values.tolist())
    labels.remove(baseline_name)
    labels.append(baseline_name)

    dataset = [(df[df[hue] == x]['n_drops'].values.tolist(
    ), df[df[hue] == x]['test_accuracy'].values.tolist()) for x in labels]
    for label, dataset in zip(labels, dataset):
        x, y = dataset[0], dataset[1]
        if label == baseline_name:
            plt.semilogx(x, y)
        else:
            plt.semilogx(x, y, marker='o')
    plt.legend(labels)

    plt.xlabel('# of instances removed')
    plt.ylabel('Accuracy')
    # save
    fig_save_dir = os.path.join(output_dir, fig_save_dir_name)
    ensure_dir(fig_save_dir)
    plt.savefig(os.path.join(fig_save_dir, 'result.png'))


def analyze(file_dir, baseline_name='No Removal'):
    # analyze and save result
    summary_df = analyze_result(os.path.join(file_dir, 'score'), baseline_name)
    summary_dir = os.path.normpath(os.path.join(file_dir, 'summary'))
    ensure_dir(summary_dir)
    summary_df.to_csv(os.path.join(summary_dir, 'summary.csv'))
    # plot result
    plot_result(summary_df, file_dir, baseline_name)
