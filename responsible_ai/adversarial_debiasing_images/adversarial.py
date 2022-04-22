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
import pickle
import numpy as np
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
from nnabla.models.imagenet import ResNet50
from sklearn.metrics import average_precision_score
from utils import utils


def clf_resnet50(layer, n_classes=1, train=True):
    """
    This function uses ResNet-50 pretrained on ImageNet as the base architecture
    and replaces the linear layer in ResNet with two linear layers with the
    hidden layer of size 2,048. Dropout and ReLU are applied
    between these. mainly this function use to classify the target network.
    """
    layer_1 = F.relu(PF.affine(layer, 2048, name='classifier_1'))
    if train:
        layer_1 = F.dropout(layer_1, 0.5)
    out = PF.affine(layer_1, n_classes, name='classifier_2')
    return layer_1, out


def domain_network(features, n_classes):
    """
    This function uses to classify the domain network.
    """
    with nn.parameter_scope('domainnetwork'):
        out2 = PF.affine(features, n_classes, name='domain')
    return out2


class adversarial:
    """
    This class is use to train the Adversarial model.

    This class uses ResNet-50 pretrained on ImageNet as the base architecture
    and replaces the linear layer in ResNet with two linear layers with the
    hidden layer of size 2,048. Dropout and ReLU are applied
    between these.
    The adversarial debiasing procedure takes inspiration from GANs (Goodfellow et al. 2014)
    for training a fair classifier. In GANs they introduce a system of two neural networks in which
    the two neural networks compete with each other to become more accurate in their predictions.
    Likewise, in adversarial debiasing, we build two classifier models
    1. Target Classifier
    2. Domain Classifier
    """

    def __init__(self, batch_size=32, learning_rate=1e-4, max_iter=5086,
                 total_epochs=50, training_ratio=3, lamda=5,
                 monitor_path=None, validation_weight=None, model_load_path=None):
        """
        Construct all the necessary attributes for the adversarial model.
        Args:
            batch_size (int): number of samples contained in each generated batch
            learning_rate (float) : learning rate
            max_iter (int) : maximum iterations for an epoch
            total_epochs (int) : total epochs to train the model
            training_ratio (int) : training ratio b/w the classifier and adversary n/w
            lamda (int) : lambda parameter weighs the adversarial loss of each class
            validation_weight : sample weights
            monitor_path (str) : model parameter to be saved
            model_load_path (str) : load the model
        """
        self.batch_size = batch_size
        self.total_epochs = total_epochs
        self.max_iter = max_iter
        self.training_ratio = training_ratio
        self.lamda = lamda
        self.acc = 0
        self.monitor_path = monitor_path
        self.validation_weight = validation_weight

        model = ResNet50()
        self.input_image = nn.Variable((self.batch_size,) + (3, 224, 224))
        self.label = nn.Variable([self.batch_size, 1])
        # fine tuning
        pool = model(self.input_image, training=True, use_up_to='pool')
        self.feature, self.clf = clf_resnet50(pool, n_classes=1)
        self.clf.persistent = True
        # loss
        self.loss = F.mean(F.sigmoid_cross_entropy(self.clf, self.label))
        self.loss.persistent = True
        # hyper parameters
        self.solver = S.Adam(learning_rate)
        self.solver.set_parameters(nn.get_parameters())

        self.domain_label = nn.Variable([self.batch_size, 1])
        self.domain_output = domain_network(self.feature, 2)
        self.domain_output.persistent = True
        self.domain_loss = F.mean(F.softmax_cross_entropy(
            self.domain_output, self.domain_label))
        self.domain_loss.persistent = True
        self.log_softmax = F.log_softmax(self.domain_output, axis=1)
        self.confusion_loss = -F.mean(self.log_softmax)
        self.confusion_loss.persistent = True

        self.adv_loss = self.loss + self.lamda * self.confusion_loss
        self.adv_loss.persistent = True
        self.domain_solver = S.Adam(learning_rate)

        with nn.parameter_scope("domainnetwork"):
            self.domain_solver.set_parameters(nn.get_parameters())

        self.x_v = nn.Variable((self.batch_size,) + (3, 224, 224))
        pool_v = model(self.x_v, training=False, use_up_to='pool')
        self.feature_v, self.v_clf = clf_resnet50(pool_v, train=False)
        self.v_clf_out = F.sigmoid(self.v_clf)

        if model_load_path is not None:
            _ = nn.load_parameters(model_load_path)

    def train(self, train_loader, val_loader):
        """
        train the adversarial model
        Args:
             train_loader (NNabla data iterator) : specify training dataset
             val_loader (NNabla data iterator) : specify validation dataset
        """

        model_save = True
        print_freq = 100
        for epoch in range(self.total_epochs):
            if epoch % self.training_ratio == 0:
                self.feature.need_grad = True
                self.clf.need_grad = True
                self.domain_output.need_grad = False
            else:
                self.feature.need_grad = False
                self.clf.need_grad = False
                self.domain_output.need_grad = True

            for i in range(self.max_iter):
                image, targets = train_loader.next()
                class_target = targets[:, 0].reshape(-1, 1)
                domain_target = targets[:, -1].reshape(-1, 1)
                self.input_image.d = image
                self.label.d = class_target
                self.domain_label.d = domain_target
                if epoch % self.training_ratio == 0:
                    self.loss.forward(clear_no_need_grad=True)
                    self.solver.zero_grad()
                    self.adv_loss.forward(clear_no_need_grad=True)
                    self.adv_loss.backward(clear_buffer=True)
                    self.solver.update()
                else:
                    self.domain_solver.zero_grad()
                    self.domain_loss.forward(clear_no_need_grad=True)
                    self.domain_loss.backward(clear_buffer=True)
                    self.domain_solver.update()
                if print_freq and (i % print_freq == 0):
                    print('Training epoch {}: [{}|{}], loss:{}, domain loss {}'.format(
                        epoch, i + 1, self.max_iter, self.loss.d, self.domain_loss.d), flush=True)

            if model_save and (epoch % self.training_ratio == 0) and (epoch > 1):
                acc, val_targets, val_scores = self.check_avg_precision(
                    val_loader, weights=self.validation_weight)
                if not os.path.exists(self.monitor_path):
                    os.makedirs(self.monitor_path)
                # computing model fairness
                cal_thresh = utils.calibrated_threshold(val_targets[:, 0],
                                                        val_scores)
                _, f1_thresh = utils.get_f1_threshold(val_targets[:, 0],
                                                      val_scores)
                val_pred = np.where(val_scores > cal_thresh, 1, 0)
                average_precision = utils.get_average_precision(val_targets[:, 0],
                                                                val_scores)
                deo = utils.get_difference_equality_opportunity(val_targets[:, 1],
                                                                val_targets[:, 0], val_pred)
                bias_amplification = utils.get_bias_amplification(val_targets[:, 1],
                                                                  val_targets[:, 0], val_pred)
                kl_divergence = utils.get_kl_divergence(val_targets[:, 1],
                                                        val_targets[:, 0], val_scores)
                val_results = {
                    'AP': average_precision,
                    'DEO': deo,
                    'BA': bias_amplification,
                    'KL': kl_divergence,
                    'f1_thresh': f1_thresh,
                    'cal_thresh': cal_thresh,
                    'epoch': epoch,
                    'accuracy': acc
                }
                if acc > self.acc:
                    self.acc = acc
                    path_best_acc_model = "{}/best_acc.h5".format(
                        self.monitor_path)
                    nn.save_parameters(path_best_acc_model)
                    print("val_results : ", val_results)
                    with open(self.monitor_path + '/val_results.pkl', 'wb+') as handle:
                        pickle.dump(val_results, handle)
                    with open(self.monitor_path + '/val_scores.pkl', 'wb+') as handle:
                        pickle.dump(val_scores, handle)
                    with open(self.monitor_path + '/val_targets.pkl', 'wb+') as handle:
                        pickle.dump(val_targets, handle)
                else:
                    path_adv_model = "{}/adversarial_model.h5".format(
                        self.monitor_path)
                    nn.save_parameters(path_adv_model)
                    with open(self.monitor_path + '/val_results_adv.pkl', 'wb+') as handle:
                        pickle.dump(val_results, handle)
                    with open(self.monitor_path + '/val_scores_adv.pkl', 'wb+') as handle:
                        pickle.dump(val_scores, handle)
                    with open(self.monitor_path + '/val_targets_adv.pkl', 'wb+') as handle:
                        pickle.dump(val_targets, handle)

    def check_avg_precision(self, loader, weights=None, print_out=True):
        """
        Compute average precision (AP) from prediction scores.
        Args:
            loader (NNabla data iterator) : specify the data iterator which to compute the AP
            weights : sample weights
            print_out (bool) : print the logs
        Return:
            acc (float) : accuracy of the model
            y_all (numpy.ndarray) : actual all target labels
            pred_all (numpy.ndarray) : predicted score
        """
        y_all, pred_all = self.get_scores(loader)
        acc = average_precision_score(
            y_all[:, 0], pred_all, sample_weight=weights)
        if print_out:
            print('Avg precision all = {}'.format(acc))
        return acc, y_all, pred_all

    def get_scores(self, loader):
        """
        get predicted scores of the  model
        Args:
            loader : specify the data loader which to predicted the scores
        Returns:
            y_all (numpy.ndarray) : actual all target labels
            pred_all (numpy.ndarray) : all predicted score

        """
        # only for validation and test set
        max_iter = int(loader.size / self.batch_size)
        rem_iter = 0
        if loader.size % self.batch_size != 0:
            rem_iter += 1
            rem_values = loader.size % self.batch_size
        y_all = []
        scores_all = []
        for _ in range(max_iter):
            image, targets = loader.next()
            self.x_v.d = image
            self.v_clf_out.forward(clear_buffer=True)
            y_all.append(targets)
            scores_all.append(self.v_clf_out.d.squeeze())
        for _ in range(rem_iter):
            loader._reset()
            image, targets = loader.next()
            self.x_v.d = image
            self.v_clf_out.forward(clear_buffer=True)
            y_all.append(targets[:rem_values])
            scores_all.append(self.v_clf_out.d.squeeze()[:rem_values])

        y_all = np.concatenate(y_all)
        pred_all = np.concatenate(scores_all)
        return y_all, pred_all
