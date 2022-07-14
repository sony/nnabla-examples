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

import os
from warnings import warn
import numpy as np
from sklearn import metrics
import nnabla as nn
from nnabla.ext_utils import get_extension_context
import classifier as clf
import data_loader as di
from utils import utils
import args


def reject_option_classification(y_true, y_predicted_score,
                                 privileged_group, metric_name="DPD",
                                 metric_upper_bound=0.10, metric_lower_bound=0.05):
    """
    Reject option classification is a postprocessing technique that swaps outcomes
    between privileged and underprivileged groups near the decision boundary.
    Args:
        y_true (numpy.ndarray) : ground truth (correct) target values.
        y_predicted_score (numpy.ndarray) : estimated probability predictions (targets scores)
                                            as returned by a classifier.
        privileged_group (numpy.ndarray): list of privileged group values.
        metric_name (str) : name of the metric to use for the optimization
                            (DPD(Demographic parity difference,
                            AAOD (Absolute average odds difference),
                            EOD (Equal opportunity difference))
        metric_upper_bound (float) : upper bound of constraint on the metric value
        metric_lower_bound (float) : upper bound of constraint on the metric value

    Returns:
        ROC_margin (float): critical region boundary,
        classification_threshold (float) : optimal classification threshold
    """

    low_classification_threshold = 0.01  # smallest classification threshold
    high_classification_threshold = 0.99  # highest classification threshold
    # number of classification threshold b/w low class threshold and high class threshold
    number_classification_threshold = 100
    # number of relevant ROC margins to be used in the optimization search
    number_ROC_margin = 50

    fair_metric_array = np.zeros(
        number_classification_threshold * number_ROC_margin)
    balanced_accuracy_array = np.zeros_like(fair_metric_array)
    ROC_margin_array = np.zeros_like(fair_metric_array)
    classification_threshold_array = np.zeros_like(fair_metric_array)
    count = 0
    # Iterate through class thresholds
    for class_thresh in np.linspace(low_classification_threshold,
                                    high_classification_threshold,
                                    number_classification_threshold):

        classification_threshold = class_thresh
        if class_thresh <= 0.5:
            low_ROC_margin = 0.0
            high_ROC_margin = class_thresh
        else:
            low_ROC_margin = 0.0
            high_ROC_margin = (1.0 - class_thresh)

        # Iterate through ROC margins
        for ROC_margin in np.linspace(
                low_ROC_margin,
                high_ROC_margin,
                number_ROC_margin):
            ROC_margin = ROC_margin
            # Predict using the current threshold and margin
            y_pred = predict(y_predicted_score, classification_threshold,
                             ROC_margin, privileged_group)
            acc = metrics.accuracy_score(y_true, y_pred)
            dpd, eod, aaod = utils.get_fairness(
                y_true, privileged_group, y_pred)
            ROC_margin_array[count] = ROC_margin
            classification_threshold_array[count] = classification_threshold
            balanced_accuracy_array[count] = acc
            if metric_name == "DPD":
                fair_metric_array[count] = dpd
            elif metric_name == "AAOD":
                fair_metric_array[count] = aaod
            elif metric_name == "EOD":
                fair_metric_array[count] = eod
            count += 1

    rel_inds = np.logical_and(fair_metric_array >= metric_lower_bound,
                              fair_metric_array <= metric_upper_bound)
    if any(rel_inds):
        best_ind = np.where(balanced_accuracy_array[rel_inds]
                            == np.max(balanced_accuracy_array[rel_inds]))[0][0]
    else:
        warn("Unable to satisy fairness constraints")
        rel_inds = np.ones(len(fair_metric_array), dtype=bool)
        best_ind = np.where(fair_metric_array[rel_inds]
                            == np.min(fair_metric_array[rel_inds]))[0][0]

    ROC_margin = ROC_margin_array[rel_inds][best_ind]
    classification_threshold = classification_threshold_array[rel_inds][best_ind]

    return ROC_margin, classification_threshold


def predict(y_predicted_score, classification_threshold, ROC_margin, privileged_group):
    """
    Obtain fair predictions with ROC method
    Args:
        y_predicted_score (numpy.ndarray): estimated probability predictions
                                           (targets score) as returned by a classifier.
        classification_threshold (float) : optimal classification threshold
        ROC_margin (float) : critical region boundary
        privileged_group (numpy.ndarray) : list of privileged group values.
    Returns:
        y_pred numpy.ndarray: predictions using ROC method.
    """

    y_pred = np.where(y_predicted_score > classification_threshold, 1, 0)

    # Indices of critical region around the classification boundary
    critical_region_indices = np.logical_and(
        y_predicted_score <= classification_threshold + ROC_margin,
        y_predicted_score > classification_threshold - ROC_margin)

    # Indices of privileged and unprivileged groups
    cond_priv = privileged_group
    cond_unpriv = privileged_group == False
    y_pred[np.logical_and(critical_region_indices,
                          cond_priv)] = 0
    y_pred[np.logical_and(critical_region_indices,
                          cond_unpriv)] = 1

    return y_pred


if __name__ == '__main__':

    opt = args.get_args()

    ctx = get_extension_context(
        opt['context'], device_id=opt['device_id'], type_config=opt['type_config'])
    nn.set_default_context(ctx)

    batch_size = opt['batch_size']
    metric_name = opt['optimization_metric']
    metric_upper_bound = opt['metric_upper_bound']
    metric_lower_bound = opt['metric_lower_bound']
    model_to_load = os.path.join(
        opt['model_save_path'], opt['attribute'], 'best_acc.h5')
    if not os.path.exists(model_to_load):
        print(f'Provided model path : {model_to_load}')
        raise ('Provided model save path is not proper')

    nn.clear_parameters()
    attribute_classifier_model = clf.AttributeClassifier(
        model_load_path=model_to_load)
    rng = np.random.RandomState(1)
    valid_loader = di.data_iterator_celeba(
        opt['celeba_image_valid_dir'], opt['attr_path'], batch_size,
        target_attribute=opt['attribute'], protected_attribute=opt['protected_attribute'],
        rng=rng)
    test_loader = di.data_iterator_celeba(
        opt['celeba_image_test_dir'], opt['attr_path'], batch_size,
        target_attribute=opt['attribute'], protected_attribute=opt['protected_attribute'],
        rng=rng)

    val_targets, val_scores = attribute_classifier_model.get_scores(
        valid_loader)
    test_targets, test_scores = attribute_classifier_model.get_scores(
        test_loader)

    ROC_margin, classification_threshold = \
        reject_option_classification(val_targets[:, 0], val_scores, val_targets[:, 1] == 1,
                                     metric_name=metric_name, metric_upper_bound=metric_upper_bound,
                                     metric_lower_bound=metric_lower_bound)
    pred = predict(val_scores, classification_threshold,
                   ROC_margin, val_targets[:, 1] == 1)
    accuracy = metrics.accuracy_score(val_targets[:, 0], pred)
    dpd, eod, aaod = utils.get_fairness(
        val_targets[:, 0], val_targets[:, 1] == 1, pred)
    utils.plot_fairness_multi(dpd, eod, aaod, accuracy, bar_x_axis="ROC")
    print(f"After applying ROC (Validation set) : \n\
                                    Demographic Parity Difference (DPD) : {round(dpd * 100, 2)} \n\
                                    Equal Opportunity Difference (EOD) : {round(eod * 100, 2)}\n\
                                    Absolute Average Odd Difference (AAOD): {round(aaod * 100, 2)}\n\
                                    Accuracy : {round(accuracy * 100, 2)}")

    pred = predict(test_scores, classification_threshold,
                   ROC_margin, test_targets[:, 1] == 1)
    accuracy = metrics.accuracy_score(test_targets[:, 0], pred)
    dpd, eod, aaod = utils.get_fairness(
        test_targets[:, 0], test_targets[:, 1] == 1, pred)
    utils.plot_fairness_multi(dpd, eod, aaod, accuracy, bar_x_axis="ROC")
    print(f"After applying ROC (Test set) : \n\
                                    Demographic Parity Difference (DPD) : {round(dpd * 100, 2)} \n\
                                    Equal Opportunity Difference (EOD) : {round(eod * 100, 2)}\n\
                                    Absolute Average Odd Difference (AAOD): {round(aaod * 100, 2)}\n\
                                    Accuracy : {round(accuracy * 100, 2)}")
