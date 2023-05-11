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

import os
import shutil
from collections import Counter

import nnabla as nn
import nnabla.experimental.parametric_function_classes as PFC
import nnabla.functions as F
import nnabla.initializer as I
import nnabla.parametric_functions as PF
import numpy as np
import scipy
from nnabla.utils.image_utils import imread, imsave
from nnabla_diffusion.config.python.datasetddpm import DatasetDDPMConfig
from nnabla_diffusion.ddpm_segmentation.data_util import (get_class_names,
                                                          get_palette)
from nnabla_diffusion.ddpm_segmentation.utils import colorize_mask, to_labels
from PIL import Image


class pixel_classifier(object):
    def __init__(self, conf: DatasetDDPMConfig):
        self.conf = conf
        self.dim = conf.dim
        self.numpy_class = conf.number_class
        self.middle_dim = [256, 128] if self.numpy_class > 30 else [128, 32]
        self.bn = conf.use_bn
        self.model_num = conf.model_num

    def classifier(self, x, t, i, recompute=False, test=False):
        with nn.parameter_scope(f"mlp_time_{t}_model_{i}"):
            with nn.parameter_scope("fc1"), nn.recompute(recompute):
                h = PF.affine(x, self.middle_dim[0])
                h = F.relu(h)
                if self.bn:
                    h = PF.batch_normalization(h, batch_stat=not test)

            with nn.parameter_scope("fc2"), nn.recompute(recompute):
                h = PF.affine(h, self.middle_dim[1])
                h = F.relu(h)
                if self.bn:
                    h = PF.batch_normalization(h, batch_stat=not test)

            with nn.parameter_scope("fc3"), nn.recompute(recompute):
                h = PF.affine(h, self.numpy_class)

        return h

    def build_training_graph(self, x, label, t, i):
        pred = self.classifier(x, t, i)
        loss = F.mean(F.softmax_cross_entropy(pred, label))
        return pred, loss

    def __call__(self, x):
        return self.classifier(x)

    def init_weights(self, init_type="normal", gain=0.02):
        """
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        """

        pass

    def predict_labels(self, conf, t, x, test=True):
        mean_seg = None
        all_seg = []
        seg_mode_ensemble = []
        for i in range(conf.datasetddpm.model_num):
            with nn.auto_forward():
                pred = self.classifier(x, t, i, test=test)
                softmax = F.softmax(pred)
            all_seg.append(pred)
            if mean_seg is None:
                mean_seg = softmax
            else:
                mean_seg += softmax

            img_seg = np.argmax(softmax.d, 1)
            img_seg = img_seg.reshape(self.dim[:-1])

            seg_mode_ensemble.append(img_seg)

        mean_seg = mean_seg / len(all_seg)

        img_seg_final = np.stack(seg_mode_ensemble, axis=-1)
        img_seg_final = scipy.stats.mode(img_seg_final, 2)[0]
        img_seg_final = img_seg_final.reshape(self.dim[:-1])
        return img_seg_final

    def load_ensemble(self, load_dir):
        suffix = "_".join([str(step) for step in self.conf.steps])

        for i in range(self.model_num):
            model_path = os.path.join(
                load_dir, "t_" + suffix + "model_" + str(i) + ".h5")

            nn.load_parameters(model_path)


def save_predictions(conf, preds, gts, imgs, editname):
    palette = get_palette(conf.datasetddpm.category)
    os.makedirs(os.path.join(conf.datasetddpm.output_dir,
                "predictions"), exist_ok=True)

    for i, (img, pred, gt) in enumerate(zip(imgs, preds, gts)):
        mask_gt = colorize_mask(gt, palette)
        mask = colorize_mask(pred, palette)
        label_convert = to_labels(
            np.expand_dims(mask, 0), palette
        )

        imsave(
            os.path.join(conf.datasetddpm.output_dir, "predictions",
                         f"batch_{i}_" + editname + "_img.png"), np.squeeze(img).astype(
                np.uint8)
        )
        imsave(
            os.path.join(conf.datasetddpm.output_dir, "predictions",
                         f"batch_{i}_" + editname + "_mask.png"), mask[:, :, ::-1]
        )


def compute_iou(conf, preds, gts, print_per_class_ious=True):
    class_names = get_class_names(conf.datasetddpm.category)
    ids = range(conf.datasetddpm.number_class)

    unions = Counter()
    intersections = Counter()

    for pred, gt in zip(preds, gts):
        for target_num in ids:
            if target_num == conf.datasetddpm.ignore_label:
                continue
            preds_tmp = (pred == target_num).astype(int)
            gts_tmp = (gt == target_num).astype(int)
            unions[target_num] += (preds_tmp | gts_tmp).sum()
            intersections[target_num] += (preds_tmp & gts_tmp).sum()

    ious = []
    for target_num in ids:
        if target_num == conf.datasetddpm.ignore_label:
            continue
        iou = intersections[target_num] / (1e-8 + unions[target_num])
        ious.append(iou)
        if print_per_class_ious:
            print(f"IOU for {class_names[target_num]} {iou:.4}")
    return np.array(ious).mean()
