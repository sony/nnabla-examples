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
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.functions as F

from nnabla.models.imagenet import ResNet50
from sklearn.metrics import average_precision_score
from utils import utils


def clf_resnet50(layer, n_classes=1, train=True):
    """This function uses ResNet-50 pretrained on ImageNet as the base architecture
    and replaces the linear layer in ResNet with two linear layers with the
    hidden layer of size 2,048. Dropout and ReLU are applied between these.
    """
    layer_1 = F.relu(PF.affine(layer, 2048, name='classifier_1'))
    if train:
        layer_1 = F.dropout(layer_1, 0.5)
    out = PF.affine(layer_1, n_classes, name='classifier_2')
    return out


class AttributeClassifier:
    """This class uses ResNet-50 pretrained on ImageNet as the base architecture
    and replaces the linear layer in ResNet with two hidden layers of size 2,048.
    Dropout and ReLU are applied.
    """

    def __init__(self, batch_size=32, learning_rate=1e-4, max_iter=5086,
                 total_epochs=20, eta=0, monitor_path=None,
                 validation_weight=None, model_load_path=None):
        """
        Construct all the necessary attributes for the attribute classifier.
        Args:
            batch_size (int): number of samples contained in each generated batch
            learning_rate (float) : learning rate
            max_iter (int) : maximum iterations for an epoch
            total_epochs (int) : total epochs to train the model
            eta (float): parameter for prejudice remover(eta)
            validation_weight : sample weights
            monitor_path (str) : model parameter to be saved
            model_load_path (str) : load the model
        """
        self.batch_size = batch_size
        self.eta = eta

        # ResNet-50 training graph
        model = ResNet50()
        self.input_image = nn.Variable((self.batch_size,) + model.input_shape)
        self.label = nn.Variable([self.batch_size, 1])
        # fine tuning
        pool = model(self.input_image, training=True, use_up_to='pool')
        self.clf = clf_resnet50(pool, n_classes=1)
        self.clf.persistent = True
        self.clf_sigm = F.sigmoid(self.clf)
        # loss
        self.loss = F.mean(F.sigmoid_cross_entropy(self.clf, self.label))
        self.loss.persistent = True
        # hyper parameters
        self.solver = S.Adam(learning_rate)
        self.solver.weight_decay(1e-05)
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

    def pr_loss(self, output_f, output_m, eta):
        """Prejudice Remover Regularizer
        Args:
            output_f (NNabla Variable): output of unprivileged class
            output_m (NNabla Variable) : output of privileged class
            eta : parameter for prejudice remover(eta)
        Returns:
            Prejudice Index
        """
        N_female = F.constant(output_f.shape[0])
        N_male = F.constant(output_m.shape[0])
        # male sample, #female sample
        Dxisi = F.stack(N_male, N_female, axis=0)
        y_pred_female = F.sum(output_f)
        y_pred_male = F.sum(output_m)
        P_ys = F.stack(y_pred_male, y_pred_female, axis=0) / Dxisi
        P = F.concatenate(output_f, output_m, axis=0)
        P_y = F.sum(P) / (output_f.shape[0]+output_m.shape[0])
        P_s1y1 = F.log(P_ys[1]) - F.log(P_y)
        P_s1y0 = F.log(1-P_ys[1]) - F.log(1-P_y)
        P_s0y1 = F.log(P_ys[0]) - F.log(P_y)
        P_s0y0 = F.log(1-P_ys[0]) - F.log(1-P_y)
        P_s1y1 = F.reshape(P_s1y1, (P_s1y1.d.size,))
        P_s1y0 = F.reshape(P_s1y0, (P_s1y0.d.size,))
        P_s0y1 = F.reshape(P_s0y1, (P_s0y1.d.size,))
        P_s0y0 = F.reshape(P_s0y0, (P_s0y0.d.size,))
        # PI
        PI_s1y1 = output_f * P_s1y1
        PI_s1y0 = (1 - output_f) * P_s1y0
        PI_s0y1 = output_m * P_s0y1
        PI_s0y0 = (1 - output_m) * P_s0y0
        PI = F.sum(PI_s1y1) + F.sum(PI_s1y0) + F.sum(PI_s0y1) + F.sum(PI_s0y0)
        PI = eta * PI
        return PI

    def train(self, train_loader, val_loader):
        """Train the attribute classifier and save model with best accuracy on validation set
        Args:
             train_loader (NNabla data iterator) : specify training dataset
             val_loader (NNabla data iterator) : specify validation dataset
        """

        model_save = True
        print_freq = 100
        for epoch in range(self.total_epochs):
            for i in range(self.max_iter):
                image, targets = train_loader.next()
                protected_variable = targets[:, 1].reshape(-1, 1)
                targets = targets[:, 0].reshape(-1, 1)
                self.input_image.d = image
                self.label.d = targets
                self.solver.zero_grad()
                self.clf_sigm.forward(clear_no_need_grad=True)
                self.output_f = self.clf_sigm[protected_variable == 0]
                self.output_m = self.clf_sigm[protected_variable == 1]
                self.pi_Loss = self.pr_loss(
                    self.output_f, self.output_m, self.eta)
                self.t_loss = self.loss + self.pi_Loss
                self.t_loss.forward(clear_no_need_grad=True)
                self.t_loss.backward(clear_buffer=True)
                self.solver.update()

                if print_freq and (i % print_freq == 0):
                    print(f'Training epoch {epoch}: [{i+1}|{self.max_iter}], Total loss:{self.t_loss.d} Loss : {self.loss.d}',
                          flush=True)
            if model_save:
                acc, val_targets, val_scores = self.check_avg_precision(
                    val_loader)
                if acc > self.acc:
                    self.acc = acc
                    if not os.path.exists(self.monitor_path):
                        os.makedirs(self.monitor_path)
                    path_best_acc_model = f"{self.monitor_path}/best_acc.h5"
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
                    deo = utils.get_diff_in_equal_opportunity(val_targets[:, 1],
                                                              val_targets[:, 0], val_pred)
                    bias_amplification = utils.get_bias_amplification(val_targets[:, 1],
                                                                      val_targets[:, 0], val_pred)
                    outf = val_scores[val_targets[:, 1] == 0]
                    outm = val_scores[val_targets[:, 1] == 1]
                    cv_score = utils.get_cvs(outf, outm, cal_thresh)
                    val_results = {
                        'AP': average_precision,
                        'CV_score': cv_score,
                        'DEO': deo,
                        'BA': bias_amplification,
                        'f1_thresh': f1_thresh,
                        'cal_thresh': cal_thresh,
                        'accuracy': acc,
                    }
                    with open(self.monitor_path + '/val_results.pkl', 'wb+') as handle:
                        pickle.dump(val_results, handle)

    def check_avg_precision(self, loader, weights=None, print_out=True):
        """Compute average precision (AP) from prediction scores.
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
            print(f'Avg precision all = {acc}')
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
