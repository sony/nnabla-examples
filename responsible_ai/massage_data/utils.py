# Copyright 2023 Sony Group Corporation.
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


import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix


def get_num_pos_neg(X, target_attribute, target_class_cond, sensitive_attribute, sensitive_class_cond):
    """

    Args:
        X (pandas DataFrame): input dataset
        target_attribute: target attribute of the dataset
        target_class_cond (int): target class label (1/0)
        sensitive_attribute (string): sensitive attribute of the dataset
        sensitive_class_cond (int): sensitive class condition (1/0)

    Returns:
        number of instances in the dataset where the target attribute satisfies the given sensitive attribute condition.

    """
    return len(X[(X[sensitive_attribute] == sensitive_class_cond) & (X[target_attribute] == target_class_cond)])


def get_num_instances(X, attribute, condition):
    """
    get the number of instances
    Args:
        X: input dataset
        attribute (string): attribute for which the number of instances needs to be counted
        condition (int): condition that the attribute must satisfy

    Returns:
        number of instances in the dataset where the attribute satisfies the given condition.

    """
    return len(X[(X[attribute] == condition)])


def get_base_rate(X, target_attribute, target_class_cond, sensitive_attribute, sensitive_class_cond):
    """
    Compute the base rate : Pr(Y = 1) = P/(P+N)
    Args:
        X (pandas DataFrame): input dataset
        target_attribute (string): target attribute of the dataset
        target_class_cond (int): target class label (1/0)
        sensitive_attribute (string): sensitive attribute of the dataset
        sensitive_class_cond (int): sensitive class condition (1/0)
    """
    return (get_num_pos_neg(X, target_attribute, target_class_cond, sensitive_attribute, sensitive_class_cond)
            / get_num_instances(X, sensitive_attribute, sensitive_class_cond))


def statistical_parity_difference(X, target_attribute, target_class_cond,
                                  sensitive_attribute, privileged_class, unprivileged_class):
    """
    Compute difference in the metric between unprivileged and privileged groups.
    Args:
        X (pandas DataFrame): input dataset
        target_attribute (string): target attribute of the dataset
        target_class_cond (int): target class label
        sensitive_attribute (string): sensitive attribute of the dataset
        privileged_class (int): privileged class label of the sensitive attribute
        unprivileged_class (int): unprivileged class label of the sensitive attribute
    Returns: Disparate Impact

    """

    positive_privileged = get_base_rate(
        X, target_attribute, target_class_cond, sensitive_attribute, privileged_class)
    positive_unprivileged = get_base_rate(X, target_attribute, target_class_cond, sensitive_attribute,
                                          unprivileged_class)

    return positive_unprivileged - positive_privileged


def disparate_impact(X, target_attribute, target_class_cond,
                     sensitive_attribute, privileged_class, unprivileged_class):
    """
    Compute the ratio of the metric between unprivileged and privileged groups.

    Args:
        X (pandas DataFrame): input dataset
        target_attribute (string): target attribute of the dataset
        target_class_cond(int): target class label
        sensitive_attribute (string): sensitive attribute of the dataset
        privileged_class (int): privileged class label of the sensitive attribute
        unprivileged_class (int): unprivileged class label of the sensitive attribute

    Returns: Disparate Impact

    """

    positive_privileged = get_base_rate(
        X, target_attribute, target_class_cond, sensitive_attribute, privileged_class)
    positive_unprivileged = get_base_rate(X, target_attribute, target_class_cond, sensitive_attribute,
                                          unprivileged_class)

    return positive_unprivileged / positive_privileged


def get_model_fairness(y, privileged_group, preds):
    """
    Get the model fairness

    Args:
        y: True labels for the data
        privileged_group: privileged group for which we want to check fairness
        preds: predicted probabilities for the positive class

    Returns:
        demographic parity(dpd),disparate_impact(di),
        equal opportunity difference(eod),
        equalised odds(AAOD)

    """
    y_unprivileged, preds_unprivileged = y[privileged_group == False], preds[
        privileged_group == False]
    y_privileged, preds_privileged = y[privileged_group], preds[privileged_group]
    cm_unprivileged = confusion_matrix(y_unprivileged, preds_unprivileged)
    cm_privileged = confusion_matrix(y_privileged, preds_privileged)

    unprivileged_PR = (
        cm_unprivileged[1, 1] + cm_unprivileged[0, 1]) / cm_unprivileged.sum()
    privileged_PR = (cm_privileged[1, 1] +
                     cm_privileged[0, 1]) / cm_privileged.sum()
    dpd = unprivileged_PR - privileged_PR
    di = unprivileged_PR / privileged_PR
    unprivileged_TPR = cm_unprivileged[1, 1] / cm_unprivileged[1].sum()
    privileged_TPR = cm_privileged[1, 1] / cm_privileged[1].sum()
    eod = unprivileged_TPR - privileged_TPR

    unprivileged_FPR = cm_unprivileged[0, 1] / cm_unprivileged[0].sum()
    privileged_FPR = cm_privileged[0, 1] / cm_privileged[0].sum()
    aaod = 0.5 * (
            abs(unprivileged_FPR - privileged_FPR) + abs(unprivileged_TPR - privileged_TPR))
    return dpd, di, eod, aaod


def train_model(clf, x_train, y_train, **kwargs):
    """
    Train the model on given training data
    Args:
        clf: various ML models
        x_train: Training data
        y_train: Training targets
        kwargs: dictionary that maps each keyword to the value that we pass alongside it
    Returns:
        classifier model
    """
    # Instantiate the classification model
    if "sample_weight" in kwargs:
        clf.fit(x_train, y_train.ravel(),
                sample_weight=kwargs["sample_weight"])
    else:
        clf.fit(x_train, y_train)
    return clf


def plot_fairness_comp(original, mitigated, metric="DPD"):
    """
    Args:
        original: original bias of the dataset
        mitigated: mitigated bias of the dataset
        metric: name of the fairness metric

    Returns:
        plot the fairness comparison

    """
    plt.figure(facecolor='#FFFFFF', figsize=(4, 4))
    plt.axhline(y=0.0, color='r', linestyle='-')
    plt.bar(["Original", "Mitigated"], [
            original, mitigated], color=["blue", "green"])
    plt.ylabel(metric)
    plt.title("Before vs After Bias Mitigation", fontsize=15)
    y = [original, mitigated]
    for index, value in enumerate(y):
        if value < 0:
            plt.text(index, value - 0.01,
                     str(round(value, 3)), fontweight='bold', color='red',
                     bbox=dict(facecolor='red', alpha=0.4))
        else:
            plt.text(index, value + 0.01,
                     str(round(value, 3)), fontweight='bold', color='red',
                     bbox=dict(facecolor='red', alpha=0.4))
    plt.grid(None, axis="y")
    plt.show()
