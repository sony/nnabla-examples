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
from sklearn.metrics import average_precision_score
import numpy as np
import nnabla as nn
import nnabla.solvers as S
import nnabla.functions as F
from nnabla.models.imagenet import ResNet50
import nnabla.parametric_functions as PF
from utils import utils


def clf_resnet50(layer, n_classes=1, train=True):
    """
    This function uses ResNet-50 pretrained on ImageNet as the base architecture
    and replaces the linear layer in ResNet with two linear layers with the
    hidden layer of size 2,048. Dropout and ReLU are applied
    between these
    """
    layer_1 = F.relu(PF.affine(layer, 2048, name='classifier_1'))
    if train:
        layer_1 = F.dropout(layer_1, 0.5)
    out = PF.affine(layer_1, n_classes, name='classifier_2')
    return out


class attribute_classifier:
    """
    This class uses ResNet-50 pretrained on ImageNet as the base architecture
    and replaces the linear layer in ResNet with two linear layers with the
    hidden layer of size 2,048. Dropout and ReLU are applied
    between these
    """

    def __init__(self, batch_size=32, learning_rate=1e-4, max_iter=5086,
                 total_epochs=50, monitor_path=None,
                 validation_weight=None, model_load_path=None):
        """
        Construct all the necessary attributes for the attribute classifier.
        Args:
            batch_size (int): number of samples contained in each generated batch
            learning_rate (float) : learning rate
            max_iter (int) : maximum iterations for an epoch
            total_epochs (int) : total epochs to train the model
            validation_weight : sample weights
            monitor_path (str) : model parameter to be saved
            model_load_path (str) : load the model
        """
        self.batch_size = batch_size
        # Resnet 50
        # training graph
        model = ResNet50()
        self.input_image = nn.Variable((self.batch_size,) + model.input_shape)
        self.label = nn.Variable([self.batch_size, 1])
        # fine tuning
        pool = model(self.input_image, training=True, use_up_to='pool')
        self.clf = clf_resnet50(pool, n_classes=1)
        self.clf.persistent = True
        # loss
        self.loss = F.mean(F.sigmoid_cross_entropy(self.clf, self.label))
        # hyper parameters
        self.solver = S.Adam(learning_rate)
        self.solver.set_parameters(nn.get_parameters())

        # validation graph
        self.x_v = nn.Variable((self.batch_size,) + model.input_shape)
        pool_v = model(self.x_v, training=False, use_up_to='pool')
        self.v_clf = clf_resnet50(pool_v, train=False)
        self.v_clf_out = F.sigmoid(self.v_clf)
        self.validation_weight = validation_weight
        # val params
        self.acc = 0.0
        self.total_epochs = total_epochs
        self.max_iter = max_iter
        self.monitor_path = monitor_path
        if model_load_path is not None:
            _ = nn.load_parameters(model_load_path)

    def train(self, train_loader, val_loader):
        """
        train the attribute classifier and save model with best accuracy on validation set
        Args:
             train_loader (NNabla data iterator) : specify training dataset
             val_loader (NNabla data iterator) : specify validation dataset
        """

        model_save = True
        print_freq = 100
        for epoch in range(self.total_epochs):
            for i in range(self.max_iter):
                image, targets = train_loader.next()
                targets = targets[:, 0].reshape(-1, 1)
                self.input_image.d = image
                self.label.d = targets
                self.solver.zero_grad()
                self.loss.forward(clear_no_need_grad=True)
                self.loss.backward(clear_buffer=True)
                self.solver.update()

                if print_freq and (i % print_freq == 0):
                    print('Training epoch {}: [{}|{}], loss:{}'.format(
                        epoch, i + 1, self.max_iter, self.loss.d), flush=True)
            if model_save:
                acc, val_targets, val_scores = \
                    self.check_avg_precision(
                        val_loader, weights=self.validation_weight)
                if acc > self.acc:
                    self.acc = acc
                    if not os.path.exists(self.monitor_path):
                        os.makedirs(self.monitor_path)
                    path_best_acc_model = "{}/best_acc.h5".format(
                        self.monitor_path)
                    nn.save_parameters(path_best_acc_model)

                    # computing model fairness
                    cal_thresh = utils.calibrated_threshold(val_targets[:, 0],
                                                            val_scores)
                    _, f1_thresh = utils.get_f1_threshold(val_targets[:, 0],
                                                          val_scores)
                    val_pred = np.where(val_scores > cal_thresh, 1, 0)
                    with open(self.monitor_path + '/val_scores.pkl', 'wb+') as handle:
                        pickle.dump(val_scores, handle)
                    with open(self.monitor_path + '/val_targets.pkl', 'wb+') as handle:
                        pickle.dump(val_targets, handle)
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
                    with open(self.monitor_path + '/val_results.pkl', 'wb+') as handle:
                        pickle.dump(val_results, handle)

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
        max_iter = int(loader.size/self.batch_size)
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
