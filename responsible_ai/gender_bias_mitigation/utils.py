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


import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GERMAN_CREDIT_DATSET_COLUMN_NAMES = ['status', 'month', 'credit_history',
                                     'purpose', 'credit_amount', 'savings', 'employment',
                                     'investment_as_income_percentage', 'personal_status',
                                     'other_debtors', 'residence_since', 'property', 'age',
                                     'installment_plans', 'housing', 'number_of_credits',
                                     'skill_level', 'people_liable_for', 'telephone',
                                     'foreign_worker', 'credit']


def load_dataset(filepath, column_names=GERMAN_CREDIT_DATSET_COLUMN_NAMES):
    """
    Loads and returns dataset loaded as pandas dataframe
    """
    try:
        dataframe = pd.read_csv(
            filepath, sep=' ', header=None, names=column_names)
    except IOError as err:
        print("IOError: {}".format(err))
        sys.exit(1)
    return dataframe


def preprocess_dataset(dataframe, label_name, protected_attribute_names=["sex"],
                       privileged_classes=[["male"]], favorable_class=1, categorical_features=None,
                       features_to_drop=None):
    """
    Dataset specific preprocessing

    Args:
        dataframe: DataFrame to perform standard preprocessing
        label_name: Name of the target variable
        protected_attribute_names: list of names corresponding to protected attribute
        privileged_classes: Each element is a list of values -> privileged classes
        favorable_class: Label values which are considered favorable
        categorical_features: to be expanded into one-hot vectors
        features_to_drop: column names to drop

    Returns:
        pandas.DataFrame: pre-processed dataframe
    """
    # Create a one-hot encoding of the categorical variables.
    categorical_features = sorted(set(categorical_features) - set(features_to_drop),
                                  key=dataframe.columns.get_loc)
    dataframe = pd.get_dummies(
        dataframe, columns=categorical_features, prefix_sep='=')

    # Map protected attributes to privileged/unprivileged
    for attr, vals in zip(protected_attribute_names, privileged_classes):
        privileged_values = [1.]
        unprivileged_values = [0.]
        if callable(vals):
            dataframe[attr] = dataframe[attr].apply(vals)
        else:
            priv = np.logical_or.reduce(
                np.equal.outer(vals, dataframe[attr].to_numpy()))
            dataframe.loc[priv, attr] = privileged_values[0]
            dataframe.loc[~priv, attr] = unprivileged_values[0]

    # map favourable class
    if callable(favorable_class):
        dataframe[label_name] = dataframe[label_name].apply(favorable_class)
    else:
        dataframe[label_name] = (dataframe[label_name]
                                 == favorable_class).astype(int)

    # Drop unrequested columns
    dataframe = dataframe[sorted(set(dataframe.columns.tolist()) - set(features_to_drop),
                                 key=dataframe.columns.get_loc)]

    return dataframe


def train_model_and_get_test_results(model, x_train, y_train, x_test, y_test, **kwargs):
    """
    Train the model on given training data and evaluate its performance on test set

    Args:
        model: various ML models
        x_train: Training data
        y_train: Training targets
        x_test: Testing data
        y_test: Testing targets
        kwargs: dictionary that maps each keyword to the value that we pass alongside it

    Returns:
        Accuracy and predictions made by the specified model
    """
    # Instantiate the classification model
    if "sample_weight" in kwargs:
        model.fit(x_train, y_train.ravel(),
                  sample_weight=kwargs["sample_weight"])
    else:
        model.fit(x_train, y_train)

    predicted = model.predict(x_test)

    # compute the accuracy of the model
    accuracy = model.score(x_test, y_test.ravel())

    return accuracy, predicted


def visualize_model_comparison(ml_models, base_fairness, mitigated_fairness, base_accuracy,
                               mitigated_accuracy):
    """
    Graphical Comparison of fairness & performance

    Args:
        ml_models: list the name of the ML models
        base_fairness: list the fairness values of the ML models, before bias mitigation
        mitigated_fairness: list the fairness values of the ML models, after bias mitigation
        base_accuracy: list the accuracy values of the ML models, before bias mitigation
        mitigated_accuracy: list the fairness values of the ML models, after bias mitigation

    """

    def autolabel(rects, axes):
        for rect in rects:
            h = rect.get_height()
            if h < 0:
                axes.text(rect.get_x() + rect.get_width() / 2., h - 0.04, '%.2f' % float(h),
                          ha='center', va='bottom')
            else:
                axes.text(rect.get_x() + rect.get_width() / 2., 1.02 * h, '%.2f' % float(h),
                          ha='center', va='bottom')

    ind = np.arange(len(ml_models))  # the x locations for the groups
    width = 0.27  # the width of the bars
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    fairness_base_bar = ax1.bar(ind, base_fairness, width, color='r',
                                label='Before bias mitigation')
    fairness_mitigated_bar = ax1.bar(ind + width, mitigated_fairness, width, color='g',
                                     label='After bias mitigation')
    ax1.axhline(y=0.0, color='r', linestyle='-')
    ax1.set_xlabel("ML models")
    ax1.set_ylabel('Fairness(SPD)')
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_xticks(ind + width)
    ax1.set_xticklabels(ml_models)
    ax1.title.set_text('(a) Fairness of ML model')
    ax1.legend()
    autolabel(fairness_base_bar, axes=ax1)
    autolabel(fairness_mitigated_bar, axes=ax1)

    base_accuracy_bar = ax2.bar(ind, base_accuracy, width, color='r',
                                label='Before bias mitigation')
    mitigated_accuracy_bar = ax2.bar(ind + width, mitigated_accuracy, width, color='g',
                                     label='After bias mitigation')
    ax2.set_xlabel("ML models")
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0.0, 1.0)
    ax2.set_xticks(ind + width)
    ax2.set_xticklabels(ml_models)
    ax2.title.set_text('(b) Performance of ML model')
    ax2.legend()
    autolabel(base_accuracy_bar, axes=ax2)
    autolabel(mitigated_accuracy_bar, axes=ax2)
    plt.show()
