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

import os
import numpy as np
import nnabla as nn
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from nnabla.ext_utils import get_extension_context
from nnabla.utils.load import load


def get_context(device_id):
    # for cli app use
    try:
        context = 'cudnn'
        ctx = get_extension_context(context, device_id=device_id)
    except (ModuleNotFoundError, ImportError):
        context = 'cpu'
        ctx = get_extension_context(context, device_id=device_id)
    # for nnc use
    config_filename = 'net.nntxt'
    if os.path.isfile(config_filename):
        config_info = load([config_filename])
        ctx = config_info.global_config.default_context
    return ctx


def red_blue_map():
    colors = []
    for i in np.linspace(1, 0, 100):
        colors.append((30. / 255, 136. / 255, 229. / 255, i))
    for i in np.linspace(0, 1, 100):
        colors.append((255. / 255, 13. / 255, 87. / 255, i))
    return LinearSegmentedColormap.from_list("red_transparent_blue", colors)


def get_model_layers(model_graph):
    model_layers = dict()
    for k, v in model_graph.variables.items():
        if ("/W" in k) or ("/b" in k):
            continue
        else:
            model_layers[k] = v
    return model_layers


def get_middle_layer(model_layers, interim_layer_index):
    return list(model_layers.values())[interim_layer_index]


def gradient(model_layers, input_layer, target_layer, output_layer, inputs, idx):

    for v in model_layers.values():
        v.grad.zero()
        v.need_grad = True
    selected = output_layer[:, idx]
    input_layer.d = inputs
    selected.forward()
    selected.backward()
    grad = target_layer.g.copy()
    return grad


def get_interim_input(input_layer, middle_layer, output_layer, inputs):
    input_layer.d = inputs
    output_layer.forward()
    middle_layer_d = middle_layer.d.copy()
    return middle_layer_d


def shap(model_graph, X, label, interim_layer_index, num_samples,
         dataset, batch_size, num_epochs=1):
    input_layer = list(model_graph.inputs.values())[0]
    model_layers = get_model_layers(model_graph)
    middle_layer = get_middle_layer(model_layers, interim_layer_index)
    output_layer = list(model_graph.outputs.values())[0]
    # get data
    if len(X.shape) == 3:
        batch_size = 1
    else:
        batch_size = len(X)

    x = nn.Variable((batch_size,) + input_layer.shape)
    # set training True since gradient of variable is necessary for SHAP
    if interim_layer_index == 0:
        data = X.reshape((1,) + X.shape)
    else:
        data = get_interim_input(input_layer, middle_layer, output_layer, X)

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
                    input_layer, middle_layer, output_layer, samples_input[j][k])[0]

        grads = []

        for b in range(0, num_samples, batch_size):
            batch_last = min(b + batch_size, num_samples)
            batch = samples_input[j][b:batch_last].copy()
            grads.append(gradient(model_layers, input_layer,
                         middle_layer, output_layer, batch, label))
        grad = [np.concatenate([g for g in grads], 0)]
        samples = grad[0] * samples_delta[0]
        phis[0][j] = samples.mean(0)

    output_phis.append(phis[0])
    return output_phis


def visualize(X, output_phis, output, ratio_num=10):
    import matplotlib
    matplotlib.use('Agg')
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
    ax.imshow(shap_plot, cmap=red_blue_map(), vmin=min_border, vmax=max_border)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output)
    fig.clf()
    plt.close()


def shap_computation(model_graph, X, label, interim_layer_index, num_samples,
                     dataset, batch_size, output):
    ctx = get_context(0)
    nn.set_default_context(ctx)
    output_phis = shap(model_graph, X, label, interim_layer_index, num_samples,
                       dataset, batch_size)
    visualize(X, output_phis, output)
