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
from sklearn.metrics import average_precision_score, f1_score, recall_score
from scipy.ndimage import gaussian_filter


def get_all_attr():
    """
    get all attributes in celebA dataset
    """
    return ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
            'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
            'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
            'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
            'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
            'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
            'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
            'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']


def compute_class_weight(loader):
    """
    Compute sample weights
    """

    cp = 0
    cn = 0
    cn_dn = 0
    cn_dp = 0
    cp_dn = 0
    cp_dp = 0

    weights = []
    max_iter = int(loader.size/32)
    for i in range(max_iter):
        _, targets = loader.next()
        class_label = targets[:, 0]
        domain_label = targets[:, 1]
        cp += class_label.sum()  # class is positive
        cn += (targets.shape[0] - class_label.sum())  # class is negative
        # class is negative, domain is negative
        cn_dn += ((class_label + domain_label) == 0).sum()
        cn_dp += ((class_label - domain_label) == -1).sum()
        cp_dn += ((class_label - domain_label) == 1).sum()
        cp_dp += ((class_label + domain_label) == 2).sum()
    for i in range(max_iter):
        _, targets = loader.next()
        class_label = targets[:, 0]
        domain_label = targets[:, 1]

        weights.append(
            (class_label*cp + (1-class_label)*cn) /
            (2*(
                    (1-class_label)*(1-domain_label)*cn_dn
                    + (1-class_label)*domain_label*cn_dp
                    + class_label*(1-domain_label)*cp_dn
                    + class_label*domain_label*cp_dp
                   )
             )
        )
    weights = np.concatenate(weights)
    return weights


def get_bias_amplification_attribute_task(running_labels, running_preds):
    """
    compute bias amplification for attribute-task

    References:
        [1] Directional Bias Amplification(https://arxiv.org/abs/2102.12594)
        [2] Fair Attribute Classification through Latent Space
        De-biasing(https://arxiv.org/abs/2012.01469)
    Args:
        running_labels : labels on test set
        running_preds : predictions on test set
    Returns:
         bias amplification
    """
    bog_tilde = np.zeros((2, 2))
    bog_gt_g = np.zeros((2, 2))
    for i, objs in enumerate([running_labels, running_preds]):
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
        domain (numpy.ndarray) : actual protected attribute
        targets (numpy.ndarray) : actual target label
        pred (numpy.ndarray) : predicted label
    Returns:
        Bias amplification
    """

    test_labels = np.zeros((targets.shape[0], 2))
    test_labels[:, 0] = targets
    test_labels[:, 1] = domain
    test_pred = np.zeros((targets.shape[0], 2))
    test_pred[:, 0] = pred
    test_pred[:, 1] = domain
    auc_bias = get_bias_amplification_attribute_task(test_labels, test_pred)
    return auc_bias


def get_difference_equality_opportunity(domain, targets, pred):
    """
    Compute the absolute difference in FNR between the protected attribute group.

    Args:
        domain (numpy.ndarray) : actual protected attribute
        targets (numpy.ndarray) : actual target label
        pred (numpy.ndarray) : predicted label
    Returns:
        difference_in_equla_opportunity(float)

    """

    g0 = np.argwhere(domain == 0)
    g1 = np.argwhere(domain == 1)
    deo = np.abs(
        (1-recall_score(targets[g0], pred[g0]))-(1-recall_score(targets[g1], pred[g1])))

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
        thresh = 0.1*t
        curr_scores = np.where(scores_all > thresh, 1, 0)
        acc = f1_score(targets_all, curr_scores)
        if acc > best_acc:
            best_acc = acc
            best_t = thresh
    one_dec = best_t

    for t in range(1, 20):
        thresh = (one_dec-0.1) + 0.01*t
        curr_scores = np.where(scores_all > thresh, 1, 0)
        acc = f1_score(targets_all, curr_scores)
        #print(thresh, acc, best_acc, flush=True)
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


def get_kl(p, q):
    """
    Kullback-Leibler divergence D(P || Q) for discrete distributions
    Args:
         p,q (float array): Discrete probability distributions.
    Returns:
        kl
    """
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def smoothed_hist_kl_distance(a, b, nbins=50, sigma=1):
    """
    smooth two histograms with Standard kernel (gaussian kernel with mean = 0 ,sigma=1),
    and calculate the KL distance between two smoothed histograms.
    Args:
        a (numpy.ndarray) : numpy nd array
        b (numpy.ndarray) : numpy nd array
        nbins (int) : number of histogram bins
        sigma : standard deviation for Gaussian kernel
    Returns:
        kl distance
    """
    ahist = np.histogram(a, bins=nbins)[0]
    bhist = np.histogram(b, bins=nbins)[0]

    asmooth = gaussian_filter(ahist, sigma)
    bsmooth = gaussian_filter(bhist, sigma)

    asmooth = asmooth/asmooth.sum() + 1e-6
    bsmooth = bsmooth/bsmooth.sum() + 1e-6

    return get_kl(asmooth, bsmooth), get_kl(bsmooth, asmooth)


def get_kl_divergence(domain, targets, scores):
    """
    Measures the divergence between score distributions (KL/threshold-invariant metric)
    References:
        [1] Towards Threshold Invariant Fair Classification(https://arxiv.org/abs/2102.12594)
        [2] Fair Attribute Classification through Latent Space De-biasing(https://arxiv.org/abs/2012.01469)
    Args:
        domain (numpy.ndarray) : actual protected attribute
        targets (numpy.ndarray) : actual target label
        scores (numpy.ndarray) : predicted scores
    Returns:
        kl_divergence
    """

    nbin = 50  # Number of histogram bins

    # a_b, b_a = smoothed_hist_kl_distance(scores[domain == 0], scores[domain == 1], nbins=nbin)
    a_b_pos, b_a_pos = smoothed_hist_kl_distance(scores[np.logical_and(domain == 0, targets == 1)],
                                                 scores[np.logical_and(domain == 1, targets == 1)], nbins=nbin)
    a_b_neg, b_a_neg = smoothed_hist_kl_distance(scores[np.logical_and(domain == 0, targets == 0)],
                                                 scores[np.logical_and(domain == 1, targets == 0)], nbins=nbin)

    return np.median([a_b_pos]+[a_b_neg]+[b_a_pos]+[b_a_neg])
