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

import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from IPython.display import Markdown, display
import sys
import numpy as np


def load_adult_dataset():
    """
    Loads Census Income Dataset; 
    target variable is whether income exceeds $50K/yr
    (For more info : https://archive.ics.uci.edu/ml/datasets/adult)

    Returns:
        features (pandas.DataFrame): features of the Census income dataset
        target (pandas.Series): target values of the Census income dataset
        sensitive_attributes(pandas.DataFrame): sensitive attributes values of Census income dataset
    """
    file_path = r'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
                    'marital_status', 'occupation', 'relationship', 'race', 'sex',
                    'capital_gain', 'capital_loss', 'hours_per_week', 'country', 'target']
    try:
        # Restrict races to White and Black
        input_data = (pd.read_csv(file_path, names=column_names,
                                  na_values="?", sep=r'\s*,\s*', engine='python').loc[lambda df: df['race'].isin(['White', 'Black'])]).dropna()
    except Exception as e:
        print("Error : ", e)
        sys.exit(1)

    # sensitive attributes; we identify 'race' and 'sex' as sensitive attributes
    # privileged class for race and sex is White and Male respectively
    sensitive_attribute_list = ['race', 'sex']
    sensitive_attributes = (input_data.loc[:, sensitive_attribute_list]
                            .assign(race=lambda df: (df['race'] == 'White').astype(int),
                                    sex=lambda df: (df['sex'] == 'Male').astype(int)))

    # targets; 1 when someone makes over 50k , 0 otherwise
    target = (input_data['target'] == '>50K').astype(int)

    # features; note that the 'target' and sensitive attribute columns are dropped
    features = (input_data
                .drop(columns=['target', 'race', 'sex', 'fnlwgt'])
                .pipe(pd.get_dummies, drop_first=True))

    display(Markdown(
        f"features : {features.shape[0]} samples, {features.shape[1]} attributes"))
    display(Markdown(f"targets : {target.shape[0]} samples"))
    display(Markdown(
        f"sensitives attributes : {sensitive_attributes.shape[0]} samples, {sensitive_attributes.shape[1]} attributes"))
    return features, target, sensitive_attributes


def CVS(output_f, output_m):
    """
    Calders and Verwer defined a discrimination score,
    by subtracting the conditional probability of the positive class given a sensitive value 
    from that given a non-sensitive value.
    Args:
        output_f (list): output of unprivileged class
        output_m (list) : output of privileged class
    Returns:
        CV Score (float)
    """
    yf_pred = (output_f >= 0.5)
    ym_pred = (output_m >= 0.5)
    corr_f = np.sum(yf_pred == True)
    corr_m = np.sum(ym_pred == True)
    P_y1_s1 = corr_f / output_f.shape[0]
    P_y1_s0 = corr_m / output_m.shape[0]
    CV_score = np.abs(P_y1_s0 - P_y1_s1)
    return round(CV_score.item(), 4)
    print('Calder-Verwer discrimination score: %.4f' % (CV_score.item()))


def fairness_plot(x, x_name, y_left, y_left_name, y_right, y_right_name):
    """
    Graphical Comparison of fairness & performance with respect to
    Prejudice Regularizer 
    """
    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax1.plot(x, y_left, color='b')
    ax1.set_xlabel(x_name, fontsize=16, fontweight='bold')
    ax1.set_ylabel(y_left_name, color='b', fontsize=16, fontweight='bold')
    ax1.xaxis.set_tick_params(labelsize=14)
    ax1.yaxis.set_tick_params(labelsize=14)
    ax1.set_ylim(min(y_left)-1, max(y_left)+1)

    ax2 = ax1.twinx()
    ax2.plot(x, y_right, color='r')
    ax2.set_ylabel(y_right_name, color='r', fontsize=16, fontweight='bold')
    ax2.set_ylim(0, max(y_right)+0.1)
    ax2.yaxis.set_tick_params(labelsize=14)
