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

import pickle
import numpy as np
from sklearn import svm
import args
from utils import utils

if __name__ == "__main__":

    opt = args.get_args()
    attr_list = utils.get_all_attr()

    X = pickle.load(
        open(r"{}/latent_vectors.pkl".format(opt['record_latent_vector']), 'rb'))
    g = pickle.load(open("{}/all_{}_scores.pkl".format(opt['fake_data_dir'],
                                                       attr_list[opt['protected_attribute']]), 'rb'))
    a = pickle.load(open("{}/all_{}_scores.pkl".format(opt['fake_data_dir'],
                                                       attr_list[opt['attribute']]), 'rb'))

    # as per the author's citation, first 15000 fake images are used to learn the separating hyperplane.
    # 10000 synthetic images for training & 5000 synthetic images for validation, labeled with baseline model.
    no_samples_train = 10000
    no_samples_valid = 5000

    total_no_of_samples = len(X)  # total no of samples
    total_samples_train = no_samples_train + \
        no_samples_valid  # total no of samples to train
    no_of_flipped_images = total_no_of_samples - \
        total_samples_train  # no of flipped images to generate

    # Split the data into training and validation sets.
    X_train = X[:no_samples_train, :]
    g_train = g[:no_samples_train]
    a_train = a[:no_samples_train]

    X_val = X[no_samples_train:no_samples_valid, :]
    g_val = g[no_samples_train:no_samples_valid]
    a_val = a[no_samples_train:no_samples_valid]

    # train latent space attribute classifiers ha & hg to learn the attributes of x in the training set.
    # normalize both wt & wg .

    # Fit a protected attribute classifier in the latent space using a linear SVM (max_iter=500000)
    clf_g = svm.LinearSVC(max_iter=500000)  # hg
    clf_g.fit(X_train, g_train)
    # Normalize so that w_g has norm 1.
    clf_g_norm = np.linalg.norm(clf_g.coef_)
    clf_g.coef_ = clf_g.coef_ / (clf_g_norm)  # w_g
    clf_g.intercept_ = clf_g.intercept_ / clf_g_norm  # b_g

    # Fit a target attribute classifier in the latent space using a linear SVM (max_iter=500000)
    clf_a = svm.LinearSVC(max_iter=500000)  # ha
    clf_a.fit(X_train, a_train)
    # Normalize so that w_t has norm 1.
    clf_a_norm = np.linalg.norm(clf_a.coef_)
    clf_a.coef_ = clf_a.coef_ / (clf_a_norm)  # w_a
    clf_a.intercept_ = clf_a.intercept_ / clf_a_norm  # b_a

    # derive the  x'
    # x' = x - ((2 * (w_g^T x + b_g))/(1 - (w_g^T w_a)^2))(w_g - (w_g^T w_a)w_a)

    g_perp_a = clf_g.coef_ - (np.sum(clf_g.coef_ * clf_a.coef_)) * clf_a.coef_

    g_perp_a = g_perp_a / np.linalg.norm(g_perp_a)

    cos_theta = np.sum(clf_g.coef_ * clf_a.coef_)
    # cos(theta)^2 + sin(theta)^2 = 1
    sin_theta = np.sqrt(1 - cos_theta * cos_theta)

    # generate the flipped images for remaining 1,60,000 images
    X_all = np.zeros((no_of_flipped_images, X.shape[1]))

    # For every x, find x' with flipped protected attribute
    for j in range(total_samples_train, total_no_of_samples):
        x = X[j]
        dist = np.sum(clf_g.coef_ * x) + clf_g.intercept_
        X_all[j - total_samples_train] = x - \
            ((2 * dist) / sin_theta) * g_perp_a

    with open(r"{}/latent_vectors_{}.pkl".format(opt['record_latent_vector'],
                                                 attr_list[opt['attribute']]), 'wb+') as handle:
        pickle.dump(X_all, handle)
