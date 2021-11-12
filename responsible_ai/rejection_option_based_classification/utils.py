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

import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from IPython.display import Markdown, display
import sys
from sklearn import metrics
import numpy as np


def load_adult_data():
    """
    Load and preprocess Census Income dataset

    This dataset is used to predict whether income exceeds $50K/yr
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
        # # Restrict races to White and Black
        input_data = (pd.read_csv(file_path, names=column_names,
                                  na_values="?", sep=r'\s*,\s*', engine='python').loc[lambda df: df['race'].isin(['White', 'Black'])]).dropna()
    except Exception as e:
        print("Error : ", e)
        sys.exit(1)

    # sensitive attributes; we identify 'race' and 'sex' as sensitive attributes
    # privileged class for race and sex is White and Male respectively
    sensitive_attribute_names = ['race', 'sex']
    sensitive_attributes = (input_data.loc[:, sensitive_attribute_names].assign(race=lambda df: (df['race'] == 'White').astype(int),
                                                                                sex=lambda df: (df['sex'] == 'Male').astype(int)))

    # targets; 1 when someone makes over 50k , otherwise 0
    target = (input_data['target'] == '>50K').astype(int)

    # features; note that the 'target' and sensitive attribute columns are dropped
    features = (input_data
                .drop(columns=['target', 'race', 'sex', 'fnlwgt'])
                .fillna('Unknown')
                .pipe(pd.get_dummies, drop_first=True))

    display(Markdown(
        f"features : {features.shape[0]} samples, {features.shape[1]} attributes"))
    display(Markdown(f"targets : {target.shape[0]} samples"))
    display(Markdown(
        f"sensitives attributes : {sensitive_attributes.shape[0]} samples, {sensitive_attributes.shape[1]} attributes"))
    display(
        Markdown(f"sensitives attributes names : {sensitive_attribute_names} "))

    return features, target, sensitive_attributes


def get_fairness(y, privileged_group, preds):
    """
    Compute multiple fairness metrics for the classifier:
    1. Demographic parity
    2. Equal opportunity
    3. Equalized odds
    Args:
        y : True data (or target, ground truth)
        privileged_group : list of privileged group values
        preds : data predicted (calculated, output) by your model
    Returns:
        demographic_parity_difference (float): Demographic parity
        equal_opportunity_difference (float): Equal opportunity
        average_abs_odds_difference(float): Equalized odds
    """

    y_unprivileged, preds_unprivileged = y[privileged_group ==
                                           False], preds[privileged_group == False]
    y_privileged, preds_privileged, = y[privileged_group], preds[privileged_group]
    cm_unprivileged = confusion_matrix(y_unprivileged, preds_unprivileged)
    cm_privileged = confusion_matrix(y_privileged, preds_privileged)
    unprivileged_PR = (
        cm_unprivileged[1, 1] + cm_unprivileged[0, 1]) / cm_unprivileged.sum()
    privileged_PR = (cm_privileged[1, 1] +
                     cm_privileged[0, 1]) / cm_privileged.sum()

    # compute demographic parity
    demographic_parity_difference = unprivileged_PR - privileged_PR
    unprivileged_TPR = cm_unprivileged[1, 1] / cm_unprivileged[1].sum()
    privileged_TPR = cm_privileged[1, 1] / cm_privileged[1].sum()

    # compute equal opportunity
    equal_opportunity_difference = unprivileged_TPR - privileged_TPR
    unprivileged_FPR = cm_unprivileged[0, 1] / cm_unprivileged[0].sum()
    privileged_FPR = cm_privileged[0, 1] / cm_privileged[0].sum()

    # compute Equalized odds
    average_abs_odds_difference = 0.5 * \
        (abs(unprivileged_FPR - privileged_FPR) +
         abs(unprivileged_TPR - privileged_TPR))

    return demographic_parity_difference, equal_opportunity_difference, average_abs_odds_difference


def plot_fairness(fairness, ax, metric="DPD", title="fairness metric", bar_x_axis="Original"):
    """
    plot single fairness metric
    """

    ax.set_ylim([-0.6, 0.6])
    ax.axhline(y=0.0, color='r', linestyle='-')
    ax.bar([bar_x_axis], fairness, color="blue", width=2)
    ax.set_ylabel(metric)
    ax.set_title(title, fontsize=10)

    for index, value in enumerate(fairness):
        if value < 0:
            ax.text(index, value - 0.1, str(round(value, 3)), fontweight='bold',
                    color='red', bbox=dict(facecolor='red', alpha=0.4))
        else:
            ax.text(index, value + 0.1, str(round(value, 3)), fontweight='bold',
                    color='red', bbox=dict(facecolor='red', alpha=0.4))


def plot_fairness_multi(DPD, EOD, AAOD, accuracy, bar_x_axis="original"):
    """
    plot three fairness metrics
    """
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=True)
    fig.suptitle("Model Fairness", fontsize=16)
    plot_fairness([DPD], ax=axes[0], metric="DPD",
                  title="Demographic Parity", bar_x_axis=bar_x_axis)
    plot_fairness([EOD], ax=axes[1], metric="EOD",
                  title="Equal Opportunity", bar_x_axis=bar_x_axis)
    plot_fairness([AAOD], ax=axes[2], metric="AAOD",
                  title="Equalized Odds", bar_x_axis=bar_x_axis)
    fig.text(0.92, 0.65, '\n'.join(
        ["Accuracy of the model:", f"- accuracy : {accuracy:.3f}"]), fontsize='15')
    plt.show()
