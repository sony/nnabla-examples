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

import sys
import numpy as np
import pandas as pd
import opendatasets as od
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve, roc_curve, auc


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

    dataframe = pd.read_csv(
            file_path, sep=' ', header=None, names=column_names)
    # derive the gender attribute from personal_status (you can refer the german.doc)
    status_map = {'A91': 'male', 'A93': 'male', 'A94': 'male',
                  'A92': 'female', 'A95': 'female'}
    dataframe['gender'] = dataframe['personal_status'].replace(status_map)

    # targets; 1 when someone makes good credit , otherwise 0
    target = (dataframe['credit'] == 1).astype(int)
    # features; note that the 'target' and sensitive attribute columns are dropped
    features = (dataframe
                .drop(columns=['credit', 'age', 'gender', 'personal_status'])
                .fillna('Unknown')
                .pipe(pd.get_dummies, drop_first=True))
    print("features sample", features.shape[0])
    print("features attributes", features.shape[1])
    print("target samples", target.shape[0])
    return features, target


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
    except Exception as e:
        print("Error : ", e)
        sys.exit(1)

    # targets; 1 when someone makes over 50k , otherwise 0
    target = (input_data['target'] == '>50K').astype(int)

    # features; note that the 'target' and sensitive attribute columns are dropped
    features = (input_data
                .drop(columns=['target', 'race', 'sex'])
                .pipe(pd.get_dummies, drop_first=True))
    print("features sample", features.shape[0])
    print("features attributes", features.shape[1])
    print("target samples", target.shape[0])
    return features, target


def roc_curve_value(y_label, y_pred, thresholds):
    '''
    calculate TPR and FPR
    Args:
        y_label     (list) : ground truth target values
        y_pred      (list) : estimated targets as returned by a classifier
        thresholds  (array): threshold values

    return:
        fpr (list): False Positive Rate
        tpr (list): True Positive Rate
    '''
    fpr = []
    tpr = []
    positive = sum(y_label)
    negative = len(y_label) - positive
    for thresh in thresholds:
        fp_count = 0
        tp_count = 0
        length = len(y_pred)
        for i in range(length):
            if y_pred[i] >= thresh:
                if y_label[i] == 1:
                    tp_count = tp_count + 1
                if y_label[i] == 0:
                    fp_count = fp_count + 1
        fpr.append(fp_count/float(negative))
        tpr.append(tp_count/float(positive))
    return fpr, tpr


def auc_score(x, y):
    '''
    Compute Area Under Curve
    Args:
        x (list): computed x values
        y (list): computed y values
    Return:
        auc_score (float): auc_score
    '''
    height = 0.5*(np.array(y[1:])+np.array(y[:-1]))
    width = -((np.array(x[1:])-np.array(x[:-1])))
    auc_score = (height*width).sum()
    return auc_score


def calc_precision_recall(y_label, y_pred):
    '''
    Calculate precission and recall
    Args:
        y_label     (list) : ground truth target values
        y_pred      (list) : estimated targets as returned by a classifier

    return:
        precision (float): precision value
        recall    (float): recall value
    '''

    tp_count = 0
    fp_count = 0
    fn_count = 0
    length = len(y_label)
    for i in range(length):
        if y_label[i] == y_pred[i] == 1:
            tp_count += 1
        if y_pred[i] == 1 and y_label[i] != y_pred[i]:
            fp_count += 1
        if y_pred[i] == 0 and y_label[i] != y_pred[i]:
            fn_count += 1
    try:
        precision = tp_count / (tp_count + fp_count)
    except:
        precision = 1
    try:
        recall = tp_count / (tp_count + fn_count)
    except:
        recall = 1
    return precision, recall


def cal_scores(thresholds, y_test_probs, y_test):
    '''
    calculate precision scores, recall scores
    Args:
        thresholds   (float): threshold values
        y_test_probs (list): estimated targets as returned by a classifier
        y_test       (list): target values

    Return:
        precision_scores (list): computed precision values
        recall_scores    (list): computed recall values
    '''
    precision_scores = []
    recall_scores = []
    for thresh in thresholds:
        y_test_preds = []
        y_test_preds = [1 if prob > thresh else 0 for prob in y_test_probs]
        precision, recall = calc_precision_recall(y_test, y_test_preds)
        precision_scores.append(precision)
        recall_scores.append(recall)
    return precision_scores, recall_scores


def get_roc_curve(fpr, tpr, auc):
    """
    plot ROC curve
    """
    plt.rcParams["figure.figsize"] = (10, 8)
    plt.title('Receiver Operating Characteristic curve')
    plt.plot([0, 1], ls="--", color='tan')
    plt.plot([0, 0], [1, 0], c=".7"), plt.plot([1, 1], c=".7")
    plt.plot(fpr, tpr, color='blue', label="ROC curve (area = %0.2f)" % auc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc=4),
    plt.savefig('ROC_curve.png')
    plt.show()


def get_pr_curve(precision_scores, recall_scores, Y_test, auc):
    """
    plot precision Recall curve
    """
    positive = sum(Y_test)
    baseline = positive / len(Y_test)
    plt.rcParams["figure.figsize"] = (10, 8)
    plt.title('Precision / Recall curve')
    plt.plot([0, 1], [baseline, baseline], linestyle='--',
             color='tan', label='Baseline')
    plt.plot(recall_scores, precision_scores, color='blue',
             label="PR curve (area = %0.2f)" % auc)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='center left')
    plt.savefig('Precision_Recall_curve.png')
    plt.show()


def get_roc_curve_multiclass(target_label, predicted_label, n_classes, labels):
    """
    plot ROC curve for multiclassification
    """
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    plt.rcParams["figure.figsize"] = (10, 8)
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(
            target_label[:, i], predicted_label[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot([0, 1], ls="--", color='tan')
        plt.plot([0, 0], [1, 0], c=".7")
        plt.plot([1, 1], c=".7")
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for label %i %s' % (
            roc_auc[i], i, labels[i]))
    plt.title('Receiver operating characteristic curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc=4)
    plt.savefig('roc_auc_multiclass.png')
    plt.show()


def get_pr_curve_multiclass(target_label, predicted_label, n_classes, labels):
    """
    plot precision Recall curve for multiclassification
    """
    precision = dict()
    recall = dict()
    pr_auc = dict()
    plt.rcParams["figure.figsize"] = (10, 8)
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(
            target_label[:, i], predicted_label[:, i])
        pr_auc[i] = auc(recall[i], precision[i])
        plt.plot(recall[i], precision[i], label='PR curve (area = %0.2f) for label %i %s' % (
            pr_auc[i], i, labels[i]))
    plt.title('Precision / Recall curve ')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc=3)
    plt.savefig('pr_auc_multiclass.png')
    plt.show()
