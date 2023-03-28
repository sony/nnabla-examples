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


def plot_fairness_comp(original, mitigated, metric="SPD"):
    """
    Args:
        original: original bias of the dataset
        mitigated: mitigated bias of the dataset
        metric: name of the fairness metric

    Returns:
        plot the fairness comparison

    """
    plt.figure(facecolor='#FFFFFF', figsize=(4, 4))
    plt.bar(["Original", "Mitigated"], [
            original, mitigated], color=["blue", "green"])
    plt.ylabel(metric)
    plt.title("Before vs After Bias Mitigation", fontsize=15)
    y = [original, mitigated]
    for index, value in enumerate(y):
        if value < 0:
            plt.text(index, value - 0.001,
                     str(round(value, 3)), fontweight='bold', color='red',
                     bbox=dict(facecolor='red', alpha=0.4))
        else:
            plt.text(index, value + 0.001,
                     str(round(value, 3)), fontweight='bold', color='red',
                     bbox=dict(facecolor='red', alpha=0.4))
    plt.grid(None, axis="y")
    plt.show()
