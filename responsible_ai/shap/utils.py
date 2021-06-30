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


import numpy as np
import nnabla as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def red_blue_map():
    colors = []
    for i in np.linspace(1, 0, 100):
        colors.append((30. / 255, 136. / 255, 229. / 255, i))
    for i in np.linspace(0, 1, 100):
        colors.append((255. / 255, 13. / 255, 87. / 255, i))
    return LinearSegmentedColormap.from_list("red_transparent_blue", colors)


def get_model_layers(model, inputs):
    if len(inputs.shape) == 3:
        batch_size = 1
    else:
        batch_size = len(inputs)

    x = nn.Variable((batch_size,) + model.input_shape)
    # set training True since gradient of variable is necessary for SHAP
    model_with_inputs = model(x, training=True, returns_net=True)
    model_layers = dict()
    for k, v in model_with_inputs.variables.items():
        if ("W" in k) or ("b" in k):
            continue
        else:
            model_layers[k] = v

    return model_layers


def gradient(model, inputs, idx, interim_layer_index):
    model_layers = get_model_layers(model, inputs)

    for v in model_layers.values():
        v.grad.zero()
        v.need_grad = True
    input_layer = list(model_layers.values())[-1]
    if interim_layer_index == 0:
        layer = input_layer
    else:
        layer = list(model_layers.values())[interim_layer_index]
    pred = list(model_layers.values())[-2]
    selected = pred[:, idx]
    input_layer.d = inputs
    selected.forward()
    selected.backward()
    grad = layer.g.copy()

    return grad


def get_interim_input(model, inputs, interim_layer_index):
    model_layers = get_model_layers(model, inputs)

    input_layer = list(model_layers.values())[-1]
    input_layer.d = inputs
    try:
        middle_layer = list(model_layers.values())[interim_layer_index]
    except IndexError:
        print('The interim layer should be an integer between 1 and the number of layers of the model!')
    pred = list(model_layers.values())[-2]
    pred.forward()
    middle_layer_d = middle_layer.d.copy()

    return middle_layer_d


def shap(model, X, label, interim_layer_index, num_samples,
         dataset, batch_size, num_epochs=1):
    # get data
    if interim_layer_index == 0:
        data = X.reshape((1,) + X.shape)
    else:
        data = get_interim_input(model, X, interim_layer_index)

    samples_input = [np.zeros((num_samples, ) + X.shape)]
    samples_delta = [np.zeros((num_samples, ) + data.shape[1:])]

    rseed = np.random.randint(0, 1e6)
    np.random.seed(rseed)
    phis = [np.zeros((1,) + data.shape[1:])]

    output_phis = []

    for j in range(num_epochs):
        for k in range(num_samples):
            rind = np.random.choice(len(dataset))
            t = np.random.uniform()
            im = dataset[rind]
            x = X.copy()
            samples_input[j][k] = (t * x + (1 - t) * im.copy()).copy()
            if interim_layer_index == 0:
                samples_delta[j][k] = (x - im.copy()).copy()
            else:
                samples_delta[j][k] = get_interim_input(
                                        model, samples_input[j][k],
                                        interim_layer_index)[0]

        grads = []

        for b in range(0, num_samples, batch_size):
            batch_last = min(b + batch_size, num_samples)
            batch = samples_input[j][b:batch_last].copy()
            grads.append(gradient(model, batch, label, interim_layer_index))
        grad = [np.concatenate([g for g in grads], 0)]
        samples = grad[0] * samples_delta[0]
        phis[0][j] = samples.mean(0)

    output_phis.append(phis[0])
    return output_phis


def visualize(X, output_phis, output, ratio_num=10):
    img = X.copy()
    height = img.shape[1]
    width = img.shape[2]
    ratio = ratio_num / height
    fig_size = np.array([width * ratio, ratio_num])
    fig, ax = plt.subplots(figsize=fig_size, dpi=1 / ratio)
    shap_plot = output_phis[0][0].sum(0)

    if img.max() > 1:
        img = img / 255.
    if img.shape[0] == 3:
        img_gray = (0.2989 * img[0, :, :] +
                    0.5870 * img[1, :, :] + 0.1140 * img[2, :, :])
    else:
        img_gray = img.reshape(img.shape[1:])

    abs_phis = np.abs(output_phis[0].sum(1)).flatten()
    max_border = np.nanpercentile(abs_phis, 99.9)
    min_border = -np.nanpercentile(abs_phis, 99.9)

    ax.imshow(img_gray, cmap=plt.get_cmap('gray'), alpha=0.15,
              extent=(-1, shap_plot.shape[1], shap_plot.shape[0], -1))
    im = ax.imshow(shap_plot, cmap=red_blue_map(),
                   vmin=min_border, vmax=max_border)
    ax.axis("off")

    fig.tight_layout()

    fig.savefig(output)
    fig.clf()
    plt.close()


def shap_computation(model, X, label, interim_layer_index, num_samples,
                     dataset, batch_size, output):
    output_phis = shap(model, X, label, interim_layer_index, num_samples,
                       dataset, batch_size)
    visualize(X, output_phis, output)
