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

import pandas as pd
import numpy as np


class DataPreprocessing:
    def __init__(self, df, categorical_features,
                 protected_attribute, privileged_class, target_variable, favorable_class, selected_features=[],
                 excluded_features=[], specific_data_preparation=None):
        """
        The DataPreprocessing class preprocess data by handling missing values, selecting features and handling
        categorical and protected attributes.
        Args:
            df (pandas dataframe): input dataframe
            categorical_features (list) : list of categorical features
            protected_attribute (list): list of the protected attributes
            privileged_class (list): list of privileged class for the protected attribute
            target_variable : the name of the target variable
            favorable_class : the favourable class of the target variable
            selected_features (list): list of selected features to include in the preprocessed dataset
            excluded_features (list): list of excluded features to exclude from the preprocessed dataset
            specific_data_preparation : specific data preparation function, it is used to preprocess the input dataframe
        """
        if specific_data_preparation:
            df = specific_data_preparation(df)
        self.protected_attribute = protected_attribute
        self.target_variable = target_variable
        categorical_features = sorted(
            set(categorical_features) - set(protected_attribute) -
            set(target_variable) - set(excluded_features),
            key=df.columns.get_loc)

        selected_features = selected_features or df.columns.tolist()
        selected = (set(selected_features) | set(self.protected_attribute)
                    | set(categorical_features) | set(self.target_variable))

        df = df[sorted(selected - set(excluded_features),
                       key=df.columns.get_loc)]
        self.data = df.dropna()

        self.data = pd.get_dummies(
            self.data, columns=categorical_features, prefix_sep='__')
        self.data, self.privileged_classes, self.unprivileged_classes = self.preprocess_protected_attributes(
            self.data, protected_attribute, privileged_class)
        self.data, self.favorable_label, self.unfavorable_label = self.map_favorable_class(self.data,
                                                                                           target_variable,
                                                                                           favorable_class)

    def map_favorable_class(self, df, label_name, favorable_class, favorable_label=1, unfavorable_label=0):
        """
        This function maps a given label column in pandas dataframe to a binary label column based on a given favourable
        class.
        if the favorable_class argument is callable, the function applies it to the label column.
        Otherwise,it assumes that the label column contains the binary labels and determine the favourable and
        unfavorable class based on favorable_class argument.

        Args:
            df (pandas dataframe): input dataframe
            label_name (str) : name of the label column
            favorable_class : favourable class is map the positive label. if a callable is provided, it is applied to label
            column.
            favorable_label (int) : the label value to use for the favourable class. Default is 1
            unfavorable_label (int) : the label value to use for the unfavourable class. Default is 0
        Returns:
            df (pandas dataframe): modified dataframe with the label column mapped to binary labels based on the favourable
            class
            favorable_label : label which used for the favourable class
            unfavorable_label :label used for the unfavourable class
        """
        if callable(favorable_class):
            df[label_name] = df[label_name].apply(favorable_class)
        else:
            unique_labels = set(df[label_name])
            if len(unique_labels) == 2:
                favorable_label = favorable_class[0]
                unfavorable_label = unique_labels.difference(
                    favorable_class).pop()
            else:
                pos = np.isin(df[label_name].to_numpy(), favorable_class)
                df[label_name] = np.where(
                    pos, favorable_label, unfavorable_label)
        return df, favorable_label, unfavorable_label

    def preprocess_protected_attributes(self, df, protected_attribute_names, privileged_classes, privileged_values=1,
                                        unprivileged_values=0):
        """
        Preprocess protected attribute columns in Pandas Dataframe by identifying the privileged and unprivileged class and replacing
        their values with privileged and unprivileged values respectively.

        Args:
            df (pandas.DataFrame): Dataframe containing the protected attribute columns to be processed
            protected_attribute_names (list) : list of string that representing the name of the protected attribute column
            privileged_classes(list or callable) : list of privileged classes, or callable that maps the protected
            attribute values to privileged class.
            privileged_values (float) : value to be assigned to rows with privileged class. Default is 1
            unprivileged_values (float) : value to be assigned to rows with unprivileged class. Default is 0
        Returns:
            df (pandas dataframe): modified dataframe
            privileged_attributes : privileged values
            unprivileged_attributes : unprivileged values
        """
        privileged_attributes = []
        unprivileged_attributes = []
        for attr, vals in zip(protected_attribute_names, privileged_classes):
            if callable(vals):
                df[attr] = df[attr].apply(vals)
            elif np.issubdtype(df[attr].dtype, np.number):
                privileged_values = np.asarray(vals, dtype=np.float64)
                unprivileged_values = np.asarray(
                    list(set(df[attr]).difference(vals)), dtype=np.float64)
            else:
                priv = np.isin(df[attr], vals)
                df.loc[priv, attr] = privileged_values
                df.loc[~priv, attr] = unprivileged_values

            privileged_attributes.append(privileged_values)
            unprivileged_attributes.append(unprivileged_values)
        return df, privileged_attributes, unprivileged_attributes

    def train_test_split(self, test_size=0.2, random_state=None, shuffle=True):
        """
        Split the input data into training and testing sets.
        Parameters:
            test_size (float or int): If float, should be between 0.0 and 1.0 and represent the proportion of the data
            to include in the test split.If int, represents the absolute number of test samples.
            random_state (int or None): The seed value for the random number generator (default None).
            shuffle (bool): Whether or not to shuffle the data before splitting (default True).
        Returns:
            X_train (pandas.Dataframe): features set of the training dataset
            X_test (pandas.Dataframe): features set of the testing dataset
            Y_train (pandas.Dataframe): target variable set of the training dataset
            Y_test (pandas.Dataframe): target variable set of the testing dataset
            Z_train (pandas.Dataframe): protected attribute set of the training dataset
            Z_test (pandas.Dataframe): protected attribute set of the testing dataset
        """

        if isinstance(test_size, float) and (test_size <= 0.0 or test_size >= 1.0):
            raise ValueError('Test size should be between 0.0 and 1.0')

        if isinstance(test_size, float):
            n_test = int(test_size * len(self.data))
        elif isinstance(test_size, int):
            n_test = test_size

        # Set random seed if provided
        if random_state is not None:
            np.random.seed(random_state)

        # Shuffle data if necessary
        if shuffle:
            indices = np.random.permutation(len(self.data))
            self.data = self.data.iloc[indices, :]
        sensitive_attribs = self.data.loc[:, self.protected_attribute]
        target = self.data.loc[:, self.target_variable]
        features = self.data.drop(columns=self.target_variable)

        # Split data
        X_train, X_test = features.iloc[n_test:, :], features.iloc[:n_test, :]
        Y_train, Y_test = target.iloc[n_test:, :], target.iloc[:n_test, :]
        Z_train, Z_test = sensitive_attribs.iloc[n_test:,
                                                 :], sensitive_attribs.iloc[:n_test, :]

        return X_train, X_test, Y_train, Y_test, Z_train, Z_test
