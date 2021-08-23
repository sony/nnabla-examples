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
import opendatasets as od
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
import sys


def display_log(log_dict):
    """
        Display and analyze the logs
    """

    confusion_matrix_log = \
        "Confusion matrices above show how the model performs on some test data. " \
        "We also print additional information (calculated from confusion matrices) to assess " \
        "the model fairness. For instance,\n" \
        "* The model predicted and gave favourable outcomes to "+str(log_dict['total_approvals'])+"" \
        " people. Among these individuals, "+str(log_dict['privileged_approvals'])+" belonged to " \
        "the privileged group and "+str(log_dict['unprivileged_approvals'])+" belonged to the unprivileged group.\n" \
        "* Positive rate (PR) for privileged group is " + str(log_dict['privileged_PR'])+", " \
        "and the PR for unprivileged group is " + str(log_dict['unprivileged_PR'])+" . " \
        "PR = TP /(TP+FP+TN+FN).\n" \
        "* True positive rate (TPR) for the privileged group is "+str(log_dict['privileged_TPR']) + ", " \
        "and TPR for the  unprivileged group is "+str(log_dict['unprivileged_TPR']) + " . " \
        "For the privileged group, the TPR = TP /(TP+FN).\n" \
        "* False positive rate (FPR) for the privileged group is " + str(log_dict['privileged_FPR']) + "," \
        " and the FPR for the unprivileged group is " + str(log_dict['unprivileged_FPR'])+" ." \
        "For the privileged group, TPR = FP /(FP+TN).\n\n" \
        "Let's check model fairness. In this tutorial, we explore three different types of metrics.\n" 
    display(Markdown(confusion_matrix_log))

    # demographic parity
    demographic_parity_header = \
        "##### Demographic parity: \n" \
        "* As discussed in the Demographic parity section, model is fair" \
        " if all segments of the protected class receive equal proportion/rate of positive outcomes." \
        " We refer to this proportion as `Positive Rate` of the model."
    display(Markdown(demographic_parity_header))
    plot_fairness([log_dict['DPD']], "demographic parity")
    demographic_parity_log = \
        "* As seen in the above diagram, difference between positive rates of " \
        "unprivileged and privileged group is "+str(log_dict['DPD'])+" , " \
        "which means the " + str(["privileged group gets " + str(round(abs(log_dict['DPD']*100), 3)) + ""
                                  "% more positive outcome" if log_dict['DPD'] < 0 else "unprivileged group"
                                  " gets " + str(round(abs(log_dict['DPD']*100), 3)) + "% more positive outcome"][0])+".\n" \
        "* Demographic parity difference (DPD) must be close to zero for the classifier to be fair.\n"\
        "* Model fairness is " + str(log_dict['DPD'])+" , which means "\
        + str(["the model treats individuals from the privileged and"
               "unprivileged group fairly, and satisfies the demographic parity fairness criterion."
               if log_dict['DPD'] == 0 else
               "the model treats individuals from the privileged and"
               "unprivileged group with an acceptable bias. To further improve model fairness bias must be mitigated."
               if (-0.10 < log_dict['DPD'] < 0.10) else
               "the model is not satisfied demographic parity fairness criterion ."
               " To improve model fairness, we need to mitigate(minimize) the bias."][0])
    display(Markdown(demographic_parity_log))

    # equal opportunity
    equal_opportunity_header = \
        "##### Equal opportunity : \n" \
        "* As discussed in the Equal opportunity section, model is fair, " \
        "if the proportion of people who should be selected by the model (`positives`) that are correctly " \
        "selected by the model is the same for each group. We refer to this proportion as the " \
        "true positive rate (TPR) of the model.\n"
    display(Markdown(equal_opportunity_header))
    plot_fairness([log_dict['EOD']], "Equal opportunity difference")
    equal_opportunity_log = \
        "* As seen in the above diagram the difference between positive rates of the " \
        "unprivileged and privileged group is "+str(log_dict['EOD'])+", " \
        "which means the " + str(["privileged group gets "+str(abs(log_dict['EOD']*100)) +
                                  "% more true positive rate. In other words, for privileged group,"
                                  "almost all people who should be approved are actually approved. "
                                  "For unprivileged group, if you should be approved, your chances of actually being "
                                  "approved are much lower" if log_dict['EOD'] < 0 else "unprivileged group gets"
                                  + str(abs(log_dict['EOD']*100)) +
                                  "% more true positive rate. In other words, for unprivileged group, "
                                  "almost all people who should be approved are actually approved. "
                                  "For privileged group, if you should be approved, your chances of actually "
                                  "being approved are much lower"][0])+".\n" \
        "* Equal opportunity difference (EOD) must be close to zero for the classifier to be fair.\n"\
        "* Model fairness is "+str(log_dict['EOD'])+". which means " \
        + str(["the model treats individuals from the privileged and "
               "unprivileged group equally and satisfies the EOD."
               if (log_dict['EOD'] == 0) else
               "the model treats individuals from the privileged and"
               "unprivileged group with an acceptable bias. To further improve model fairness bias must be mitigated."
               if (-0.10 < log_dict['EOD'] < 0.10) else ""
               "the model is not satisfied Equal opportunity fairness criterion ."
               " To improve model fairness , we need to mitigate (minimize) the bias."][0])
    display(Markdown(equal_opportunity_log))

    # equalized odds
    equalized_odds_header =\
        "##### Equalized odds : \n" \
        "* As discussed in the equalized section , model is fair, if the proportion of people correctly identify " \
        "the positive outcome at equal rates across groups , but also miss-classify the positive outcome " \
        "at equal rates across groups . we refer to this TPR and FPR of the model.\n"
    display(Markdown(equalized_odds_header))
    plot_fairness([log_dict['AAOD']], "Average abs odd difference")
    equalized_odds_log = \
        "* As seen in the above diagram , the average abs odd difference between privileged group" \
        " and unprivileged group is "+str(log_dict['AAOD'])+"\n" \
        "* Average abs odds difference (AAOD) must be close to zero for the classifier to be fair.\n" \
        "* Model fairness is " +str(log_dict['AAOD'])+", which means " \
        + str(["the model is correctly identify the positive outcome, "
               "and miss-classify the positive outcome at equal rates across groups . "
               "and it satisfies the equalized odd fairness criterion."
               if (log_dict['AAOD'] == 0) else
               "the model is not satisfied Equalized odd fairness criterion ."
               " To improve model fairness , we need to mitigate (minimize) the bias."][0])

    display(Markdown(equalized_odds_log))


def load_german_data():
    """
    Load and preprocess german credit card dataset and returns the feature, target, and sensitive attribute.

    This dataset classifies people described by a set of attributes as good or bad credit risks.
    (For more info : https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29)
    Returns:
        features (pandas.DataFrame): features of the german credit card dataset
        target (pandas.Series): target values of the german credit card dataset
        sensitive_attributes(pandas.DataFrame): sensitive attributes values of german credit card dataset
    """
    file_path = r'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data'
    column_names = ['status', 'month', 'credit_history',
                    'purpose', 'credit_amount', 'savings', 'employment',
                    'investment_as_income_percentage', 'personal_status',
                    'other_debtors', 'residence_since', 'property', 'age',
                    'installment_plans', 'housing', 'number_of_credits',
                    'skill_level', 'people_liable_for', 'telephone',
                    'foreign_worker', 'credit']
    try:
        dataframe = pd.read_csv(
            file_path, sep=' ', header=None, names=column_names)
    except Exception as e :
        print("Error : ", e)
        sys.exit(1)
    # derive the gender attribute from personal_status (you can refer the german.doc)
    status_map = {'A91': 'male', 'A93': 'male', 'A94': 'male',
                  'A92': 'female', 'A95': 'female'}
    dataframe['gender'] = dataframe['personal_status'].replace(status_map)

    # sensitive attributes; we identify 'age' and 'gender' as sensitive attributes
    # privileged class for age and gender is male and age >=26 respectively.
    sensitive_attribs = ['age', 'gender']
    sensitive_attributes = (dataframe.loc[:, sensitive_attribs]
                            .assign(age=lambda df: (df['age'] >= 26).astype(int),
                                    gender=lambda df: (df['gender'] == 'male').astype(int)))

    # targets; 1 when someone makes good credit , otherwise 0
    target = (dataframe['credit'] == 1).astype(int)
    # features; note that the 'target' and sensitive attribute columns are dropped
    features = (dataframe
                .drop(columns=['credit', 'age', 'gender', 'personal_status'])
                .fillna('Unknown')
                .pipe(pd.get_dummies, drop_first=True))

    display(Markdown(f"features : {features.shape[0]} samples, {features.shape[1]} attributes"))
    display(Markdown(f"targets : {target.shape[0]} samples"))
    display(Markdown(f"sensitives attributes : {sensitive_attributes.shape[0]} samples, {sensitive_attributes.shape[1]} attributes"))

    return features, target, sensitive_attributes


def load_adult_data():
    """
    Load and preprocess Census Income dataset and returns the feature, target, and sensitive attribute.

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
        # Restrict races to White and Black
        input_data = (pd.read_csv(file_path, names=column_names,
                                  na_values="?", sep=r'\s*,\s*', engine='python').loc[lambda df: df['race'].isin(['White', 'Black'])]).dropna()
    except Exception as e :
        print("Error : ", e)
        sys.exit(1)

    # sensitive attributes; we identify 'race' and 'sex' as sensitive attributes
    # privileged class for race and sex is White and Male respectively
    sensitive_attribs = ['race', 'sex']
    sensitive_attributes = (input_data.loc[:, sensitive_attribs]
                            .assign(race=lambda df: (df['race'] == 'White').astype(int),
                                    sex=lambda df: (df['sex'] == 'Male').astype(int)))

    # targets; 1 when someone makes over 50k , otherwise 0
    target = (input_data['target'] == '>50K').astype(int)

    # features; note that the 'target' and sensitive attribute columns are dropped
    features = (input_data
                .drop(columns=['target', 'race', 'sex'])
                .pipe(pd.get_dummies, drop_first=True))

    display(Markdown(f"features : {features.shape[0]} samples, {features.shape[1]} attributes"))
    display(Markdown(f"targets : {target.shape[0]} samples"))
    display(Markdown(f"sensitives attributes : {sensitive_attributes.shape[0]} samples, {sensitive_attributes.shape[1]} attributes"))
    return features, target, sensitive_attributes


def load_compas_data():
    """
    Load and preprocess ProPublica Recidivism/COMPAS dataset and returns the feature, target, and sensitive attribute.
    This dataset is used to assess the likelihood that a criminal defendant will re-offend.
    (For more info : https://github.com/propublica/compas-analysis)

    Returns:
        features (pandas.DataFrame): features of the COMPAS dataset
        target (pandas.Series): target values of the COMPAS dataset
        sensitive_attributes(pandas.DataFrame): sensitive attributes values of COMPAS dataset
    """
    file_path = r'https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv'
    try:
        # Restrict races to African-American and Caucasian
        input_data = (pd.read_csv(file_path, index_col='id').loc[lambda df: df['race'].isin(
            ['African-American', 'Caucasian'])])
    except Exception as e:
        print("Error : ", e)
        sys.exit(1)

    input_data = input_data[['sex', 'age', 'age_cat', 'race',
                             'juv_fel_count', 'juv_misd_count', 'juv_other_count',
                             'priors_count', 'c_charge_degree', 'c_charge_desc',
                             'two_year_recid']].dropna()
    # sensitive attributes; we identify 'race' and 'sex' as sensitive attributes
    # privileged class for race and sex is Caucasian and Female respectively.
    sensitive_attribs = ['race', 'sex']
    sensitive_attributes = (input_data.loc[:, sensitive_attribs]
                            .assign(race=lambda df: (df['race'] == 'Caucasian').astype(int),
                                    sex=lambda df: (df['sex'] == 'Female').astype(int)))
    # in the dataset the label value 0 is considered for no-recidivism and 1 is considered for did-recidivism
    # for shake of understanding we will map the label value 0 for did-recidivism and 1 for no-recidivism
    # targets; 1 when someone makes has no-recidivism, otherwise 0
    target = (input_data['two_year_recid'] == 0).astype(int)

    # features; note that the 'target' and sensitive attribute columns are dropped
    features = (input_data
                .drop(columns=['two_year_recid', 'race', 'sex'])
                .pipe(pd.get_dummies, drop_first=True))
    display(Markdown(f"features : {features.shape[0]} samples, {features.shape[1]} attributes"))
    display(Markdown(f"targets : {target.shape[0]} samples"))
    display(Markdown(f"sensitives attributes : {sensitive_attributes.shape[0]} samples, {sensitive_attributes.shape[1]} attributes"))

    return features, target, sensitive_attributes


def load_bank_data():
    """
    Load and preprocess Bank marketing dataset and returns the feature, target, and sensitive attribute.
    The data is related with direct marketing campaigns (phone calls) of a Portuguese banking institution.
    The classification goal is to predict if the client will subscribe a term deposit (variable y).
    (for more info : https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)

    Returns:
        features (pandas.DataFrame): features of the Bank marketing dataset
        target (pandas.Series): target values of the Bank marketing dataset
        sensitive_attributes(pandas.DataFrame): sensitive attributes values of Bank marketing dataset
    """

    try:
        http_response = urlopen(r'https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip')
        zipfile = ZipFile(BytesIO(http_response.read()))
        zipfile.extractall(path='.')
        file_path = "./bank-additional/bank-additional-full.csv"
        input_data = pd.read_csv(file_path, sep=';', na_values="unknown").dropna()
    except Exception as e:
        print("Error : ", e)
        sys.exit(1)

    # sensitive attributes; we identify "age" as sensitive attributes
    # privileged class age >=25.
    sensitive_attribs = ['age']
    sensitive_attributes = (input_data.loc[:, sensitive_attribs]
                            .assign(age=lambda df: (df['age'] >= 25).astype(int)))
    # targets; 1 if the client will subscribe a term deposit, otherwise 0
    target = (input_data['y'] == 'yes').astype(int)

    # features; note that the 'target' and sensitive attribute columns are dropped
    features = (input_data
                .drop(columns=['y', 'age'])
                .pipe(pd.get_dummies, drop_first=True))
    display(Markdown(f"features : {features.shape[0]} samples, {features.shape[1]} attributes"))
    display(Markdown(f"targets : {target.shape[0]} samples"))
    display(Markdown(f"sensitives attributes : {sensitive_attributes.shape[0]} samples, {sensitive_attributes.shape[1]} attributes"))

    return features, target, sensitive_attributes


def load_gender_discrimination_data():
    """
    Load Gender discrimination dataset and returns the feature, target, and sensitive attribute.

    Gender discrimination influence exists or not on payments and promotion,
    (for more info : https://www.kaggle.com/hjmjerry/gender-discrimination). To download this dataset
    please provide your Kaggle credentials.

    Returns:
        features (pandas.DataFrame): features of the Bank marketing dataset
        target (pandas.Series): target values of the Bank marketing dataset
        sensitive_attributes(pandas.DataFrame): sensitive attributes values of Bank marketing dataset
    """
    try:
        dataset_url = 'https://www.kaggle.com/hjmjerry/gender-discrimination'
        od.download(dataset_url)
        input_data = pd.read_csv(r'./gender-discrimination/Lawsuit.csv')
    except Exception as e:
        print("Error : ", e)
        sys.exit(1)
    # sensitive attributes; we identify "gender" as sensitive attributes
    # privileged class male.
    sensitive_attribs = ['Gender']
    sensitive_attributes = (input_data.loc[:, sensitive_attribs]
                            .assign(Gender=lambda df: (df['Gender'] == 1).astype(int)))
    # targets; Rank 1 if the faculty will get promote to full professor , otherwise 0
    target = (input_data['Rank'] == 3).astype(int)

    # features; note that the 'target' and sensitive attribute columns are dropped
    features = (input_data
                .drop(columns=['Rank','Gender'])
                .pipe(pd.get_dummies, drop_first=True))
    display(Markdown(f"features : {features.shape[0]} samples, {features.shape[1]} attributes"))
    display(Markdown(f"targets : {target.shape[0]} samples"))
    display(Markdown(f"sensitives attributes : {sensitive_attributes.shape[0]} samples, {sensitive_attributes.shape[1]} attributes"))
    return features, target, sensitive_attributes

def plot_confusion_matrix(y_true, y_pred, display_labels=["bad", "good"]):
    """
    Load Bank marketing dataset and returns the feature, target, and sensitive attribute

    Args:
        y_true: Ground truth (correct) target values.
        y_pred: Estimated targets as returned by a classifier.
        display_labels: Display labels for plot.

    Returns:
        cm: confusion matrix,
        disp: ConfusionMatrixDisplay
    """

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=display_labels)
    return cm, disp.plot(include_values=True, cmap=plt.cm.gist_ncar, xticks_rotation='horizontal', values_format='')


def plot_fairness(fairness, metric="None"):
    """
    graphical visualization of fairness of the model

    Args:
        fairness: fairness of the ML model.
        metric: name of the fairness metric.

    """
    plt.figure(facecolor='#FFFFFF', figsize=(4, 4))
    plt.ylim([-0.6, 0.6])
    plt.axhline(y=0.0, color='r', linestyle='-')
    plt.bar(["Original"], fairness, color="blue", width=2)
    plt.ylabel(metric)
    plt.title("Fairness of the model", fontsize=15)

    for index, value in enumerate(fairness):
        if value < 0:
            plt.text(index, value - 0.1,
                     str(round(value, 3)), fontweight='bold', color='red', bbox=dict(facecolor='red', alpha=0.4))
        else:
            plt.text(index, value + 0.1,
                     str(round(value, 3)), fontweight='bold', color='red', bbox=dict(facecolor='red', alpha=0.4))
    plt.show()
