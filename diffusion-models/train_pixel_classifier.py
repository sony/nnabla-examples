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

import argparse
import collections
import gc
import json
import math
import os
import sys

import hydra
import nnabla as nn
import nnabla.solvers as S
import nnabla_diffusion.config as config
import numpy as np
from neu.misc import get_current_time, init_nnabla
from nnabla.logger import logger
from nnabla_diffusion.dataset import SimpleDataIterator
from nnabla_diffusion.ddpm_segmentation.datasets import (
    ImageLabelDataIterator, PixelWiseDataIterator)
from nnabla_diffusion.ddpm_segmentation.model import FeatureExtractorDDPM
from nnabla_diffusion.ddpm_segmentation.pixel_classifier import (
    compute_iou, pixel_classifier, save_predictions)
from nnabla_diffusion.ddpm_segmentation.utils import calc_acc
from omegaconf import OmegaConf
from tqdm import tqdm


def prepare_data(model, conf, loaded_conf, comm):
    def label_creator_callback(path):
        return os.path.splitext(path)[0] + ".npy"

    data_config = config.DatasetConfig(
        batch_size=1,
        dataset_root_dir=conf.datasetddpm.training_path,
        image_size=loaded_conf.model.image_shape[:-1]
    )

    data_iterator = ImageLabelDataIterator(
        conf=data_config,
        num_images=conf.datasetddpm.training_number,
        comm=comm,
        label_creator_callback=label_creator_callback
    )

    dataset_num = data_iterator._size

    if conf.datasetddpm.share_noise:
        np.random.seed(seed=conf.datasetddpm.seed)
        noise = np.random.rand(
            1,
            conf.datasetddpm.image_size,
            conf.datasetddpm.image_size,
            3
        )
    else:
        noise = None

    X, y = [], []
    for row in tqdm(range(dataset_num)):
        image, label = data_iterator.next()
        activations = model.extract_features(image, noise=noise)
        features = model.collect_features(activations)

        d = features.d.shape[1]
        features = features.d.transpose(
            1, 0, 2, 3).reshape(d, -1).transpose(1, 0)

        for target in range(conf.datasetddpm.number_class):
            if target == conf.datasetddpm.ignore_label:
                continue
            if 0 < np.sum(label == target) < 20:
                print(
                    f"Delete small annotation from image | label {target}"
                )
                label[label == target] = conf.datasetddpm.ignore_label

        label = label.flatten()
        features = features[label != conf.datasetddpm.ignore_label]
        label = label[label != conf.datasetddpm.ignore_label]
        X.append(features)
        y.append(label)

    logger.info(f"===== total dimention: {d} ===== ")
    X = np.vstack(X)
    y = np.hstack(y)

    return X, y


def evaluation(model, conf, comm):
    def label_creator_callback(path):
        return os.path.splitext(path)[0] + ".npy"

    suffix = "_".join([str(step) for step in conf.datasetddpm.steps])
    data_config = config.DatasetConfig(
        batch_size=1,
        dataset_root_dir=conf.datasetddpm.testing_path,
        image_size=conf.datasetddpm.dim[:-1],
        shuffle_dataset=False
    )

    test_data_iterator = ImageLabelDataIterator(
        conf=data_config,
        num_images=conf.datasetddpm.testing_number,
        comm=comm,
        label_creator_callback=label_creator_callback
    )

    dataset_num = test_data_iterator._size
    if conf.datasetddpm.share_noise:
        np.random.seed(seed=conf.datasetddpm.seed)
        noise = np.random.rand(
            1,
            conf.datasetddpm.image_size,
            conf.datasetddpm.image_size,
            3
        )
    else:
        noise = None

    classifier = pixel_classifier(conf=conf.datasetddpm)
    classifier.load_ensemble(conf.datasetddpm.output_dir)

    img_paths, imgs, preds, gts = [], [], [], []
    for row in tqdm(range(dataset_num)):
        image, label = test_data_iterator.next()
        path = test_data_iterator._data_source.img_paths[row]
        imgname = os.path.splitext(os.path.basename(path))[0]
        img_paths.append(path)

        activations = model.extract_features(image, noise=noise)
        features = model.collect_features(activations)
        d = features.d.shape[1]
        features = features.d.transpose(
            1, 0, 2, 3).reshape(d, -1).transpose(1, 0)
        feature_size = len(features)
        B, D = feature_size, conf.datasetddpm.dim[-1]

        x = nn.Variable((B, D))
        x.d = features

        pred = classifier.predict_labels(conf, suffix, x, test=True)
        imgs.append(image)
        gts.append(label.squeeze())
        preds.append(pred)

        save_predictions(conf, [pred], [label.squeeze()], [
                         (image + 1) * 127.5], imgname)
    miou = compute_iou(conf, preds, gts)
    results_dict = {"miou": miou}
    with open(os.path.join(conf.datasetddpm.output_dir, "iou.json"), "w") as f:
        json.dump(results_dict, f)
    logger.info(f"==== Overall mIoU: {miou} ====")


def train(model, conf, loaded_conf, comm):
    logger.info("training start")

    suffix = "_".join([str(step) for step in conf.datasetddpm.steps])

    # if not os.path.exists("./features.npy"):
    features, labels = prepare_data(
        model=model, conf=conf, loaded_conf=loaded_conf, comm=comm)
    logger.info("==== prepared ====")

    feature_config = config.DatasetConfig(
        batch_size=conf.datasetddpm.batch_size,
        shuffle_dataset=conf.datasetddpm.shuffle_dataset
    )
    pixel_wise_iterator = PixelWiseDataIterator(
        feature_config, features, labels)

    logger.info(
        f" ==== max_label {conf.datasetddpm.number_class} ===="
    )
    logger.info(
        f" ---- Current number data {len(features)} ===="
    )

    output_dir = conf.datasetddpm.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for MODEL_NUMBER in range(conf.datasetddpm.model_num):

        gc.collect()

        classifier = pixel_classifier(conf=conf.datasetddpm)
        # classifier.init_weights()

        # build training graph
        B, D = conf.datasetddpm.batch_size, conf.datasetddpm.dim[-1]
        train_iter = math.ceil(len(features) / B)

        x = nn.Variable((B, D))
        y = nn.Variable((B, 1))
        pred, loss_train = classifier.build_training_graph(
            x, y, t=suffix, i=MODEL_NUMBER)

        classifier_param = collections.OrderedDict()
        for key, param in nn.get_parameters().items():
            if "mlp" in key and "affine" in key:
                param.grad.zero()
                classifier_param[key] = param

        solver = S.Adam()
        solver.set_parameters(classifier_param)

        break_count = 0
        best_loss = 10000000
        stop_sign = 0
        for epoch in range(100):
            print("Epoch: ", epoch)
            for iteration in range(train_iter):
                X_batch, y_batch = pixel_wise_iterator.next()
                y_batch = y_batch.reshape(len(y_batch), 1)

                x.d = X_batch
                y.d = y_batch
                loss_train.forward()
                solver.zero_grad()
                loss_train.backward()
                solver.update()

                acc = calc_acc(pred.d, y_batch)
                if iteration % 1000 == 0:
                    print(
                        "Epoch : ",
                        str(epoch),
                        "iteration",
                        iteration,
                        "loss",
                        loss_train.d,
                        "acc",
                        acc,
                    )

                if epoch > 3:
                    if float(loss_train.d) < best_loss:
                        best_loss = float(loss_train.d)
                        break_count = 0
                    else:
                        break_count += 1

                    if break_count > 50:
                        stop_sign = 1
                        print(
                            "*************** Break, Total iters,",
                            iteration,
                            ", at epoch",
                            str(epoch),
                            "***************",
                        )
                        break

            if stop_sign == 1:
                break

        model_path = os.path.join(
            output_dir, "t_" + suffix + "model_" + str(MODEL_NUMBER) + ".h5")
        print("save to: ", model_path)

        save_params = collections.OrderedDict()
        for name, param in nn.get_parameters(grad_only=False).items():
            if "mlp" in name:
                save_params[name] = param
        nn.save_parameters(model_path, save_params)


@ hydra.main(version_base=None, config_path="yaml/", config_name="config_seg_train")
def main(conf: config.TrainDatasetDDPMScriptsConfig):
    comm = init_nnabla(ext_name="cudnn",
                       device_id=conf.runtime.device_id,
                       type_config="float",
                       random_pseed=True)

    loaded_conf: config.LoadedConfig = config.load_saved_conf(
        conf.datasetddpm.config)

    # model definition
    model = FeatureExtractorDDPM(datasetddpm_conf=conf.datasetddpm,
                                 diffusion_conf=loaded_conf.diffusion,
                                 model_conf=loaded_conf.model)

    assert os.path.exists(
        conf.datasetddpm.h5), f"{conf.datasetddpm.h5} is not found. Please make sure the h5 file exists."
    nn.parameter.load_parameters(conf.datasetddpm.h5)

    output_dir = conf.datasetddpm.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    suffix = "_".join([str(step) for step in conf.datasetddpm.steps])
    pretrained = [
        os.path.exists(os.path.join(output_dir, f"t_{suffix}model_{i}.h5"))
        for i in range(conf.datasetddpm.model_num)
    ]

    if not all(pretrained):
        # train all remaining models
        train(model=model, conf=conf, loaded_conf=loaded_conf, comm=comm)
    else:
        logger.info("===== Skip Training =====")
    evaluation(model, conf, comm)


if __name__ == "__main__":
    main()
