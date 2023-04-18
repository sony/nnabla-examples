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
from zipfile import ZipFile
from urllib.request import urlopen
from . import data_process as dp
from io import BytesIO

supported_datasets = ['adult', 'german', 'compas', 'bank', "lsac"]


def default_preprocessing_german(df):
    """
    This function takes in a pandas data frame(df) containing the German dataset and performs default
    preprocessing steps.
    it derives the "sex" attribute based on the "personal_status" variable and return the modified dataframe.
    Args:
        df (pandas dataframe): data frame that containing the German credit dataset
    Returns:
        df (pandas dataframe): modified data frame after deriving the "sex" attribute based on "personal_status" variable.
    """

    status_map = {'A91': 'male', 'A93': 'male', 'A94': 'male',
                  'A92': 'female', 'A95': 'female'}
    df['sex'] = df['personal_status'].replace(status_map)

    return df


def default_preprocessing_compas(df):
    """
    This function takes in a pandas data frame(df) containing the ProPublica COMPAS dataset and performs the same
    preprocessing steps as the original analysis.
    https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb

    Args:
        df (pandas dataframe): data frame that containing the COMPAS dataset
    Returns:
        df (pandas dataframe): modified dataframe after performing the same preprocessing steps as the original
        analysis
    """

    return df[(df.days_b_screening_arrest <= 30)
              & (df.days_b_screening_arrest >= -30)
              & (df.is_recid != -1)
              & (df.c_charge_degree != 'O')
              & (df.score_text != 'N/A')]


def load_data(data_name: str, download=True, filepath=None):
    """
    Loads the dataset specified by data name using either a download url or an existing the filepath(csv).
    the function also pre-process the dataset.

    Args:
        data_name (str):name of the dataset to be loaded. must be one of the supported dataset
        download (bool): whether to download the dataset or not. Default is True.
        filepath (str) : The file path to an existing dataset (expects .csv file)
    Returns:
        data_processor (DataPreprocessing): DataPreprocessing object that contains the preprocessor dataset and relevant
        information such as protected attributes(privileged & unprivileged) and target variable.

    """

    data_name = data_name.lower()
    if data_name not in supported_datasets:
        print(f"Supported datasets are {supported_datasets}")
        raise Exception(f"{data_name} is not supported")

    if data_name == "adult":
        column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                        'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                        'hours-per-week', 'native-country', 'income']
        if download:
            train_url = r'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
            test_url = r'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
            try:
                train_samples = pd.read_csv(train_url, names=column_names, na_values="?", sep=r'\s*,\s*',
                                            engine='python')
                test_samples = pd.read_csv(test_url, names=column_names, na_values="?", sep=r'\s*,\s*', engine='python',
                                           skiprows=1)
                dataframe = pd.concat([test_samples, train_samples])
            except FileNotFoundError:
                print("The file could not be found.")
                sys.exit(1)
        else:
            print("Download is set to False. Processing existing dataset...")
            if not filepath:
                print("ERROR: Please provide a file path for the existing dataset.")
                return
            else:
                print(
                    f"Processing dataset {data_name} at file path {filepath}")
                dataframe = pd.read_csv(filepath, names=column_names, na_values="?", sep=r'\s*,\s*',
                                        engine='python')

        categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race',
                                'sex',
                                'native-country']
        protected_attr = ['sex', 'race']
        target_variable = ['income']
        privileged_classes = ['Male', 'White']
        favorable_class = ['>50K', '>50K.']
        data_processor = dp.DataPreprocessing(dataframe, categorical_features, protected_attr, privileged_classes,
                                              target_variable,
                                              favorable_class)
        return data_processor
    elif data_name == 'german':
        column_names = ['status', 'month', 'credit_history',
                        'purpose', 'credit_amount', 'savings', 'employment',
                        'investment_as_income_percentage', 'personal_status',
                        'other_debtors', 'residence_since', 'property', 'age',
                        'installment_plans', 'housing', 'number_of_credits',
                        'skill_level', 'people_liable_for', 'telephone',
                        'foreign_worker', 'credit']
        if download:
            filepath = r'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data'
            try:
                dataframe = pd.read_csv(
                    filepath, sep=' ', header=None, names=column_names)
            except FileNotFoundError:
                print("The file could not be found.")
                sys.exit(1)
            except:
                print("An error occurred while reading the file.")
                sys.exit(1)
        else:
            print("Download is set to False. Processing existing dataset...")
            if not filepath:
                print("ERROR: Please provide a file path for the existing dataset.")
                return
            else:
                print(
                    f"Processing dataset {data_name} at file path {filepath}")
                dataframe = pd.read_csv(
                    filepath, sep=' ', header=None, names=column_names)
        categorical_features = ['status', 'credit_history', 'purpose',
                                'savings', 'employment', 'other_debtors', 'property',
                                'installment_plans', 'housing', 'skill_level', 'telephone',
                                'foreign_worker']
        protected_attr = ['sex']
        target_variable = ['credit']
        privileged_classes = ['male']
        favorable_class = 1.0
        data_processor = dp.DataPreprocessing(dataframe, categorical_features, protected_attr, privileged_classes,
                                              target_variable,
                                              favorable_class, excluded_features=[
                                                  'personal_status'],
                                              specific_data_preparation=default_preprocessing_german)
        return data_processor
    elif data_name == "compas":
        if download:
            filepath = r'https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv'
            try:
                dataframe = pd.read_csv(filepath, index_col='id', na_values=[])
            except FileNotFoundError:
                print("The file could not be found.")
                sys.exit(1)
            except:
                print("An error occurred while reading the file.")
                sys.exit(1)
        else:
            print("Download is set to False. Processing existing dataset...")
            if not filepath:
                print("ERROR: Please provide a file path for the existing dataset.")
                return
            else:
                print(
                    f"Processing dataset {data_name} at file path {filepath}")
                dataframe = pd.read_csv(
                    filepath, sep=' ', header=None, names=column_names)

        categorical_features = ['age_cat', 'c_charge_degree', 'c_charge_desc']
        selected_features = ['sex', 'age', 'age_cat', 'race',
                             'juv_fel_count', 'juv_misd_count', 'juv_other_count',
                             'priors_count', 'c_charge_degree', 'c_charge_desc',
                             'two_year_recid']
        protected_attr = ['sex', 'race']
        target_variable = ['two_year_recid']
        privileged_classes = ['Female', 'Caucasian']
        favorable_class = 0.0

        data_processor = dp.DataPreprocessing(dataframe, categorical_features, protected_attr, privileged_classes,
                                              target_variable,
                                              favorable_class, selected_features=selected_features,
                                              specific_data_preparation=default_preprocessing_compas)
        return data_processor
    elif data_name == "bank":
        if download:
            try:
                http_response = urlopen(
                    r'https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip')
                zipfile = ZipFile(BytesIO(http_response.read()))
                zipfile.extractall(path='.')
                filepath = "./bank-additional/bank-additional-full.csv"
                dataframe = pd.read_csv(
                    filepath, sep=';', na_values="unknown").dropna()
            except Exception as e:
                print("Error : ", e)
                sys.exit(1)
        else:
            print("Download is set to False. Processing existing dataset...")
            if not filepath:
                print("ERROR: Please provide a file path for the existing dataset.")
                return
            else:
                print(
                    f"Processing dataset {data_name} at file path {filepath}")
                dataframe = pd.read_csv(
                    filepath, sep=';', na_values="unknown").dropna()
        categorical_features = ['job', 'marital', 'education', 'default',
                                'housing', 'loan', 'contact', 'month', 'day_of_week',
                                'poutcome']
        protected_attr = ['age']
        target_variable = ['y']
        privileged_classes = [lambda x: x >= 25]
        favorable_class = 'no'
        data_processor = dp.DataPreprocessing(dataframe, categorical_features, protected_attr, privileged_classes,
                                              target_variable,
                                              favorable_class)
        return data_processor
    elif data_name == "lsac":
        if download:
            try:
                http_response = urlopen(
                    r'http://www.seaphe.org/databases/LSAC/LSAC_SAS.zip')
                zipfile = ZipFile(BytesIO(http_response.read()))
                zipfile.extractall(path='.')
                filepath = "./lsac.sas7bdat"
                dataframe = pd.read_sas(filepath)
            except Exception as e:
                print("Error : ", e)
                sys.exit(1)
        else:
            print("Download is set to False. Processing existing dataset...")
            if not filepath:
                print("ERROR: Please provide a file path for the existing dataset.")
                return
            else:
                print(
                    f"Processing dataset {data_name} at file path {filepath}")
                dataframe = pd.read_csv(filepath)
        categorical_features = []
        protected_attr = ['gender', 'race1']
        target_variable = ['pass_bar']
        privileged_classes = [b'male', b'white']
        favorable_class = 1.0
        selected_features = ['gender', 'race1', 'lsat', 'ugpa', 'pass_bar']
        data_processor = dp.DataPreprocessing(dataframe, categorical_features, protected_attr, privileged_classes,
                                              target_variable,
                                              favorable_class, selected_features=selected_features)
        return data_processor
