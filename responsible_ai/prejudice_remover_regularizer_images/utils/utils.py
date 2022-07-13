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
from sklearn.metrics import average_precision_score, f1_score, recall_score


def get_auc_bias(gt_label, pred_label):
    """
    compute bias amplification for attribute-task
    References:
        [1] Directional Bias Amplification(https://arxiv.org/abs/2102.12594)
        [2] Fair Attribute Classification through Latent Space
        De-biasing(https://arxiv.org/abs/2012.01469)
    Args:
        gt_label: ground truth labels
        pred_label: predicated labels
    Returns:
         bias amplification
    """
    bog_tilde = np.zeros((2, 2))
    bog_gt_g = np.zeros((2, 2))
    for i, objs in enumerate([gt_label, pred_label]):
        female = np.where(objs[:, 1] == 0)[0]  # Unprivileged class
        male = np.where(objs[:, 1] == 1)[0]  # Privileged class
        kitchen = np.where(objs[:, 0] == 0)[0]  # Unfavourable class
        sports = np.where(objs[:, 0] == 1)[0]  # Favourable class
        if i == 0:
            bog_tilde[0][0] = len(set(kitchen) & set(female))
            bog_tilde[0][1] = len(set(kitchen) & set(male))
            bog_tilde[1][0] = len(set(sports) & set(female))
            bog_tilde[1][1] = len(set(sports) & set(male))
        elif i == 1:
            bog_gt_g[0][0] = len(set(kitchen) & set(female))
            bog_gt_g[0][1] = len(set(kitchen) & set(male))
            bog_gt_g[1][0] = len(set(sports) & set(female))
            bog_gt_g[1][1] = len(set(sports) & set(male))

    total_images_train = np.sum(bog_tilde)
    data_bog = bog_tilde / np.sum(bog_tilde, axis=0)
    pred_bog = bog_gt_g / np.sum(bog_tilde, axis=0)
    p_t_a = bog_tilde / np.sum(bog_tilde, axis=0)
    p_t = np.sum(bog_tilde, axis=1) / total_images_train
    diff = np.zeros_like(data_bog)
    for i in range(len(data_bog)):
        for j in range(len(data_bog[0])):
            diff[i][j] = pred_bog[i][j] - data_bog[i][j]
            indicator = np.sign(p_t_a[i][j] - p_t[i])
            if indicator == 0:
                diff[i][j] = 0
            elif indicator == -1:
                diff[i][j] = - diff[i][j]
    value = np.nanmean(diff)
    return value


def get_bias_amplification(domain, targets, pred):
    """
    BA measures how much more often a target attribute is predicted with a
    protected attribute than the ground truth value.
    References:
        [1] Directional Bias Amplification(https://arxiv.org/abs/2102.12594)
        [2] Fair Attribute Classification through Latent Space
        De-biasing(https://arxiv.org/abs/2012.01469)
    Args:
        domain (numpy.ndarray): protected attribute
        targets (numpy.ndarray): target label
        pred (numpy.ndarray): predicted label
    Returns:
        Bias amplification
    """

    test_labels = np.zeros((targets.shape[0], 2))
    test_labels[:, 0] = targets
    test_labels[:, 1] = domain
    test_pred = np.zeros((targets.shape[0], 2))
    test_pred[:, 0] = pred
    test_pred[:, 1] = domain
    auc_bias = get_auc_bias(test_labels, test_pred)
    return auc_bias


def get_diff_in_equal_opportunity(domain, targets, pred):
    """
    Compute the absolute difference in FNR between the protected attribute group.
    Args:
        domain (numpy.ndarray) : actual protected attribute
        targets (numpy.ndarray) : actual target label
        pred (numpy.ndarray) : predicted label
    Returns:
        difference_in_equal_opportunity(float)
    """

    g0 = np.argwhere(domain == 0)
    g1 = np.argwhere(domain == 1)
    deo = np.abs(
        (1 - recall_score(targets[g0], pred[g0])) - (1 - recall_score(targets[g1], pred[g1])))

    return np.median(deo)


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


def get_f1_threshold(targets_all, scores_all):
    """
    get the f1 threshold and accuracy
    Args:
        targets_all (numpy.ndarray) : actual target label
        scores : predicted scores
    Returns:
        best_acc : best accuracy
        best_t : best threshold
    """
    best_t = -1.0
    best_acc = 0.0
    for t in range(1, 10):
        thresh = 0.1 * t
        curr_scores = np.where(scores_all > thresh, 1, 0)
        acc = f1_score(targets_all, curr_scores)
        if acc > best_acc:
            best_acc = acc
            best_t = thresh
    one_dec = best_t

    for t in range(1, 20):
        thresh = (one_dec - 0.1) + 0.01 * t
        curr_scores = np.where(scores_all > thresh, 1, 0)
        acc = f1_score(targets_all, curr_scores)
        # print(thresh, acc, best_acc, flush=True)
        if acc > best_acc:
            best_acc = acc
            best_t = thresh

    return best_acc, best_t


def calibrated_threshold(targets, scores):
    """
    Calibrated threshold
    Args:
        targets (numpy.ndarray) : actual target label
        scores (numpy.ndarray) : predicted scores
    Returns:
        calibrated threshold
    """
    cp = int(targets.sum())
    scores_copy = np.copy(scores)
    scores_copy.sort()
    thresh = scores_copy[-cp]
    return thresh


def get_cvs(output_f, output_m, cal_thresh):
    """
    Calders and Verwer defined a discrimination score,
    by subtracting the conditional probability of the positive class given a sensitive value
    from that given a non-sensitive value.
    Args:
        output_f (list): output of unprivileged class
        output_m (list) : output of privileged class
        cal_thresh (float) : calibrated threshold
    Returns:
        CV Score (float)
    """
    yf_pred = (output_f >= cal_thresh)
    ym_pred = (output_m >= cal_thresh)
    corr_f = np.sum(yf_pred == True)
    corr_m = np.sum(ym_pred == True)
    P_y1_s1 = corr_f / output_f.shape[0]
    P_y1_s0 = corr_m / output_m.shape[0]
    CV_score = np.abs(P_y1_s0 - P_y1_s1)
    return round(CV_score.item(), 4)


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
            ax.text(index, value - 0.1,
                    str(round(value, 3)), fontweight='bold', color='red',
                    bbox=dict(facecolor='red', alpha=0.4))
        else:
            ax.text(index, value + 0.1,
                    str(round(value, 3)), fontweight='bold', color='red',
                    bbox=dict(facecolor='red', alpha=0.4))


def plot_fairness_multi(DEO, CV_score, BA, accuracy, bar_x_axis="original"):
    """
    plot fairness metrics
    """
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=True)
    fig.suptitle("Model Fairness", fontsize=16)
    plot_fairness([DEO], ax=axes[0], metric="DEO",
                  title="Difference in Equal opportunity(DEO)", bar_x_axis=bar_x_axis)
    plot_fairness([CV_score], ax=axes[1], metric="CV Score",
                  title="CV score", bar_x_axis=bar_x_axis)
    plot_fairness([BA], ax=axes[2], metric="BA",
                  title="Bias Amplification (BA)", bar_x_axis=bar_x_axis)

    fig.text(0.92, 0.65, '\n'.join(
        ["Average precession score:", f"- AP : {accuracy:.3f}"]), fontsize='15')
    plt.show()
