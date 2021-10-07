# Copyright (c) 2021 Sony Group Corporation. All Rights Reserved.
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
import functools
import os
from distutils.util import strtobool

import h5py
import nnabla as nn
import nnabla.functions as F
import numpy as np
from nnabla.ext_utils import get_extension_context
from nnabla.utils.data_iterator import data_iterator


from cifar10_load import Cifar10NumpySource
from model import vgg16_prediction

bs_valid = 100
CHECKPOINTS_PATH_FORMAT = "params_270.h5"


def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def feature_extractor(dataloaders_dict, image_valid, pred_hidden, pred):
    phase_feature, phase_preds = {}, {}
    for phase, dataloader in dataloaders_dict.items():
        extracted_feature, preds = [], []
        iteration = int(dataloader._size / bs_valid)
        for j in range(iteration):
            image, label = dataloader.next()
            image_valid.d = image
            pred.forward(clear_buffer=True)
            extracted_feature.append(pred_hidden.d.reshape(len(image), -1))
            preds.append(pred.d)
        concat = np.concatenate(extracted_feature, 0)
        preds = np.concatenate(preds, 0)
        phase_feature[phase] = concat
        phase_preds[phase] = preds
    return phase_feature, phase_preds


def main():
    extension_module = args.context
    ctx = get_extension_context(
        extension_module, device_id=args.device_id, type_config=args.type_config
    )
    nn.set_default_context(ctx)
    input_dir = (
        "./data/input/shuffle" if args.shuffle_label else "./data/input/no_shuffle"
    )

    X_train = np.load(os.path.join(input_dir, "x_train.npy"))
    X_val = np.load(os.path.join(input_dir, "x_val.npy"))
    if args.shuffle_label:
        print("## label shuffled")
        Y_train = np.load(os.path.join(input_dir, "y_shuffle_train.npy"))
    else:
        Y_train = np.load(os.path.join(input_dir, "y_train.npy"))
    Y_val = np.load(os.path.join(input_dir, "y_val.npy"))

    prediction = functools.partial(vgg16_prediction)

    test = True
    image_valid = nn.Variable((bs_valid, 3, 32, 32))
    output, pred_hidden = prediction(image_valid, test)
    pred_hidden.persistent = True
    pred = F.softmax(output, 1)

    save_dir = "shuffle" if args.shuffle_label else "no_shuffle"
    weight_dir = os.path.join("./tmp.monitor", save_dir)
    nn.load_parameters(os.path.join(weight_dir, CHECKPOINTS_PATH_FORMAT))

    affine_weight = []
    last_layer = [name for name in nn.get_parameters().keys()
                  ][-1].split("/")[0]
    for name, param in nn.get_parameters().items():
        if last_layer in name:
            affine_weight.append(param.d)

    data_save_dir = (
        "./data/info/shuffle" if args.shuffle_label else "./data/info/no_shuffle"
    )
    ensure_dir(data_save_dir)

    data_source_train = Cifar10NumpySource(X_train, Y_train)
    train_loader = data_iterator(
        data_source_train, bs_valid, None, False, False)
    data_source_val = Cifar10NumpySource(X_val, Y_val)
    val_loader = data_iterator(data_source_val, bs_valid, None, False, False)
    dataloaders_dict = {"train": train_loader, "test": val_loader}

    phase_feature, phase_output = feature_extractor(
        dataloaders_dict, image_valid, pred_hidden, pred
    )

    with h5py.File(os.path.join(data_save_dir, "info.h5"), "w") as hf:
        hf.create_dataset("label", data=Y_train)
        hf.create_group("param")
        for name, param in zip(["weight", "bias"], affine_weight):
            hf["param"].create_dataset(name, data=param)
        hf.create_group("feature")
        for phase, feature in phase_feature.items():
            print(feature.shape)
            hf["feature"].create_dataset(phase, data=feature)
        hf.create_group("output")
        for phase, output in phase_output.items():
            hf["output"].create_dataset(phase, data=output)

    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature generator")
    parser.add_argument("--shuffle_label", type=strtobool)
    parser.add_argument("--device_id", "-d", type=str, default="0")
    parser.add_argument("--type_config", "-t", type=str, default="float")
    parser.add_argument("--context", "-c", type=str, default="cudnn")
    args = parser.parse_args()
    main()
