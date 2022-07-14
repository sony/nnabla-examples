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

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix


def get_best_classification_thershold(val_scores, val_targets):
    """
    Compute best classfication thershold on validation data set
    Args:
        val_scores (numpy.ndarray): estimated probability predictions (targets score)
                                    as returned by a classifier on validation dataset.
        val_targets (numpy.ndarray) :  actual validation taget labels
    Returns:
        best_acc (float): best accuracy on validation dataset.
        best_classification_thershold (float): best classification thershold on validation dataset.
    """

    num_thresh = 100
    ba_arr = np.zeros(num_thresh)
    classification_thershold_array = np.linspace(0.01, 0.99, num_thresh)
    for idx, classification_thershold in enumerate(classification_thershold_array):
        y_val_pred = np.where(val_scores > classification_thershold, 1, 0)
        val_acc = metrics.accuracy_score(val_targets, y_val_pred)
        ba_arr[idx] = val_acc
    best_ind = np.where(ba_arr == np.max(ba_arr))[0][0]
    best_classification_thershold = classification_thershold_array[best_ind]
    best_acc = np.max(ba_arr)
    print("Best balanced accuracy  = %.4f" % best_acc)
    print("Optimal classification threshold = %.4f" %
          best_classification_thershold)
    return best_acc, best_classification_thershold


def get_average_precision(targets, scores):
    """
    Compute average precision score
    Args:
        targets (numpy.ndarray) : actual target label
        scores (numpy.ndarray) : predicted scores
    Returns:
        Average precission
    """

    avg_prec = average_precision_score(targets, scores)

    return np.median(avg_prec)


def get_fairness(y, privileged_group, preds):
    """
    Compute multiple fairness metrics for the classifier:
    1. Demographic parity
    2. Equal opportunity
    3. Equalized odds
    Args:
        y (numpy.ndarray): True data (or target, ground truth)
        privileged_group (numpy.ndarray): list of privileged group values
        preds (numpy.ndarray): data predicted (calculated, output) by your model
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


def plot_fairness_multi(dpd, eod, aaod, accuracy, bar_x_axis="original"):
    """
    plot three fairness metrics
    Args:
        dpd (float) : demographic parity difference
        eod (float) : equal opportunity difference
        aaod (float) : absolute average odd difference
        accuracy (float) : accuracy of the model
        bar_x_axis (str): bar name of the X-axis
    """
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=True)
    fig.suptitle("Model Fairness", fontsize=16)
    plot_fairness([dpd], ax=axes[0], metric="DPD",
                  title="Demographic Parity", bar_x_axis=bar_x_axis)
    plot_fairness([eod], ax=axes[1], metric="EOD",
                  title="Equal Opportunity", bar_x_axis=bar_x_axis)
    plot_fairness([aaod], ax=axes[2], metric="AAOD",
                  title="Equalized Odds", bar_x_axis=bar_x_axis)
    fig.text(0.92, 0.65, '\n'.join(
        ["Accuracy of the model:", f"- accuracy : {accuracy:.3f}"]), fontsize='15')
    plt.show()
