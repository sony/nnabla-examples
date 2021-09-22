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

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import lines
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib


def plot_additive_plot(data, figsize, text_rotation=0, min_perc=0.05):

    negative_features, total_neg, positive_features, total_pos = format_data(
        data)

    base_value = data['base_value']
    out_value = data['out_value']
    offset_text = (np.abs(total_neg) + np.abs(total_pos)) * 0.04

    fig, ax = plt.subplots(figsize=figsize)

    update_limits(ax, total_pos, positive_features, total_neg,
                  negative_features, base_value)

    separators = (ax.get_xlim()[1] - ax.get_xlim()[0]) / 200

    rectangle_list, separator_list = plot_bars(out_value, negative_features, 'negative',
                                               separators)
    for i in rectangle_list:
        ax.add_patch(i)

    for i in separator_list:
        ax.add_patch(i)

    rectangle_list, separator_list = plot_bars(out_value, positive_features, 'positive',
                                               separators)
    for i in rectangle_list:
        ax.add_patch(i)

    for i in separator_list:
        ax.add_patch(i)

    total_effect = np.abs(total_neg) + total_pos
    fig, ax = plot_labels(fig, ax, out_value, negative_features, 'negative',
                          offset_text, total_effect, min_perc=min_perc, text_rotation=text_rotation)

    fig, ax = plot_labels(fig, ax, out_value, positive_features, 'positive',
                          offset_text, total_effect, min_perc=min_perc, text_rotation=text_rotation)

    plot_higher_lower_element(out_value, offset_text)

    plot_base_element(base_value, ax)

    out_names = data['out_names'][0]
    plot_output_element(out_names, out_value, ax)

    if data['link'] == 'logit':
        plt.xscale('logit')
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.ticklabel_format(style='plain')
        plt.show()
    return fig


def format_data(data):
    negative_features = np.array([[data['features'][x]['effect'],
                                  data['features'][x]['value'],
                                  data['feature_names'][x]]
                                  for x in data['features'].keys() if data['features'][x]['effect'] < 0])

    negative_features = np.array(
        sorted(negative_features, key=lambda x: float(x[0]), reverse=False))

    positive_features = np.array([[data['features'][x]['effect'],
                                   data['features'][x]['value'],
                                   data['feature_names'][x]]
                                 for x in data['features'].keys() if data['features'][x]['effect'] >= 0])
    positive_features = np.array(
        sorted(positive_features, key=lambda x: float(x[0]), reverse=True))

    if data['link'] == 'identity':
        def convert_func(x): return x
    elif data['link'] == 'logit':
        def convert_func(x): return 1 / (1 + np.exp(-x))
    else:
        assert False, "ERROR: Unrecognized link function: " + str(data['link'])

    neg_val = data['out_value']
    for i in negative_features:
        val = float(i[0])
        neg_val = neg_val + np.abs(val)
        i[0] = convert_func(neg_val)
    if len(negative_features) > 0:
        total_neg = np.max(negative_features[:, 0].astype(float)) - \
                    np.min(negative_features[:, 0].astype(float))
    else:
        total_neg = 0

    pos_val = data['out_value']
    for i in positive_features:
        val = float(i[0])
        pos_val = pos_val - np.abs(val)
        i[0] = convert_func(pos_val)

    if len(positive_features) > 0:
        total_pos = np.max(positive_features[:, 0].astype(float)) - \
                    np.min(positive_features[:, 0].astype(float))
    else:
        total_pos = 0

    data['out_value'] = convert_func(data['out_value'])
    data['base_value'] = convert_func(data['base_value'])

    return negative_features, total_neg, positive_features, total_pos


def update_limits(ax, total_pos, pos_features, total_neg,
                  neg_features, base_value):
    ax.set_ylim(-0.5, 0.15)
    padding = np.max([np.abs(total_pos) * 0.2,
                      np.abs(total_neg) * 0.2])

    if len(pos_features) > 0:
        min_x = min(np.min(pos_features[:, 0].astype(
            float)), base_value) - padding
    else:
        min_x = 0
    if len(neg_features) > 0:
        max_x = max(np.max(neg_features[:, 0].astype(
            float)), base_value) + padding
    else:
        max_x = 0
    ax.set_xlim(min_x, max_x)

    plt.tick_params(top=True, bottom=False, left=False, right=False, labelleft=False,
                    labeltop=True, labelbottom=False)
    plt.locator_params(axis='x', nbins=12)

    for key, spine in zip(plt.gca().spines.keys(), plt.gca().spines.values()):
        if key != 'top':
            spine.set_visible(False)


def plot_bars(out_value, features, feature_type, separators, bar=0.1):
    rectangle_list = []
    separator_list = []

    pre_val = out_value
    for index, features in zip(range(len(features)), features):
        if feature_type == 'positive':
            left_bound = float(features[0])
            right_bound = pre_val
            pre_val = left_bound

            separator_indent = np.abs(separators)
            separator_pos = left_bound
            colors = ['#FF0D57', '#FFC3D5']
        else:
            left_bound = pre_val
            right_bound = float(features[0])
            pre_val = right_bound

            separator_indent = - np.abs(separators)
            separator_pos = right_bound
            colors = ['#1E88E5', '#D1E6FA']

        if index == 0:
            if feature_type == 'positive':
                points_rectangle = [[left_bound, 0],
                                    [right_bound, 0],
                                    [right_bound, bar],
                                    [left_bound, bar],
                                    [left_bound + separator_indent, (bar / 2)]
                                    ]
            else:
                points_rectangle = [[right_bound, 0],
                                    [left_bound, 0],
                                    [left_bound, bar],
                                    [right_bound, bar],
                                    [right_bound + separator_indent, (bar / 2)]
                                    ]

        else:
            points_rectangle = [[left_bound, 0],
                                [right_bound, 0],
                                [right_bound + separator_indent *
                                    0.90, (bar / 2)],
                                [right_bound, bar],
                                [left_bound, bar],
                                [left_bound + separator_indent * 0.90, (bar / 2)]]

        line = plt.Polygon(points_rectangle, closed=True, fill=True,
                           facecolor=colors[0], linewidth=0)
        rectangle_list += [line]

        points_separator = [[separator_pos, 0],
                            [separator_pos + separator_indent, (bar / 2)],
                            [separator_pos, bar]]

        line = plt.Polygon(points_separator, closed=None, fill=None,
                           edgecolor=colors[1], lw=3)
        separator_list += [line]

    return rectangle_list, separator_list


def plot_labels(fig, ax, out_value, features, feature_type, offset_text, total_effect=0, min_perc=0.05, text_rotation=0):
    start_text = out_value
    pre_val = out_value

    if feature_type == 'positive':
        colors = ['#FF0D57', '#FFC3D5']
        alignement = 'right'
        sign = 1
    else:
        colors = ['#1E88E5', '#D1E6FA']
        alignement = 'left'
        sign = -1

    if feature_type == 'positive':
        x, y = np.array([[pre_val, pre_val], [0, -0.18]])
        line = lines.Line2D(x, y, lw=1., alpha=0.5, color=colors[0])
        line.set_clip_on(False)
        ax.add_line(line)
        start_text = pre_val

    box_end = out_value
    val = out_value
    for feature in features:
        feature_contribution = np.abs(
            float(feature[0]) - pre_val) / np.abs(total_effect)
        if feature_contribution < min_perc:
            break

        val = float(feature[0])

        if feature[1] == "":
            text = feature[2]
        else:
            text = feature[2] + ' = ' + feature[1]

        if text_rotation != 0:
            va = 'top'
        else:
            va = 'baseline'

        textout_val = plt.text(start_text - sign * offset_text,
                               -0.15, text,
                               fontsize=12, color=colors[0],
                               horizontalalignment=alignement,
                               va=va,
                               rotation=text_rotation)
        textout_val.set_bbox(dict(facecolor='none', edgecolor='none'))

        fig.canvas.draw()
        box_size = textout_val.get_bbox_patch(
        ).get_extents().transformed(ax.transData.inverted())
        if feature_type == 'positive':
            box_end_ = box_size.get_points()[0][0]
        else:
            box_end_ = box_size.get_points()[1][0]

        if box_end_ > ax.get_xlim()[1]:
            textout_val.remove()
            break

        if (sign * box_end_) > (sign * val):
            x, y = np.array([[val, val], [0, -0.18]])
            line = lines.Line2D(x, y, lw=1., alpha=0.5, color=colors[0])
            line.set_clip_on(False)
            ax.add_line(line)
            start_text = val
            box_end = val

        else:
            box_end = box_end_ - sign * offset_text
            x, y = np.array([[val, box_end, box_end],
                             [0, -0.08, -0.18]])
            line = lines.Line2D(x, y, lw=1., alpha=0.5, color=colors[0])
            line.set_clip_on(False)
            ax.add_line(line)
            start_text = box_end

        pre_val = float(feature[0])

    extent = [out_value, box_end, 0, -0.31]
    path = [[out_value, 0], [pre_val, 0], [box_end, -0.08],
            [box_end, -0.2], [out_value, -0.2],
            [out_value, 0]]

    path = Path(path)
    patch = PathPatch(path, facecolor='none', edgecolor='none')
    ax.add_patch(patch)

    lower_lim, upper_lim = ax.get_xlim()
    if (box_end < lower_lim):
        ax.set_xlim(box_end, upper_lim)

    if (box_end > upper_lim):
        ax.set_xlim(lower_lim, box_end)

    if feature_type == 'positive':
        colors = np.array([(255, 13, 87), (255, 255, 255)]) / 255.
    else:
        colors = np.array([(30, 136, 229), (255, 255, 255)]) / 255.

    cm = matplotlib.colors.LinearSegmentedColormap.from_list('cm', colors)

    _, Z2 = np.meshgrid(np.linspace(0, 10), np.linspace(-10, 10))
    im = plt.imshow(Z2, interpolation='quadric', cmap=cm,
                    vmax=0.01, alpha=0.3,
                    origin='lower', extent=extent,
                    clip_path=patch, clip_on=True, aspect='auto')
    im.set_clip_path(patch)

    return fig, ax


def plot_output_element(out_name, out_value, ax):
    x, y = np.array([[out_value, out_value], [0, 0.24]])
    line = lines.Line2D(x, y, lw=2., color='#F2F2F2')
    line.set_clip_on(False)
    ax.add_line(line)

    font0 = FontProperties()
    font = font0.copy()
    font.set_weight('bold')
    textout_val = plt.text(out_value, 0.25, '{0:.2f}'.format(out_value),
                           fontproperties=font,
                           fontsize=14,
                           horizontalalignment='center')
    textout_val.set_bbox(dict(facecolor='white', edgecolor='white'))

    textout_val = plt.text(out_value, 0.33, out_name,
                           fontsize=12, alpha=0.5,
                           horizontalalignment='center')
    textout_val.set_bbox(dict(facecolor='white', edgecolor='white'))


def plot_base_element(base_value, ax):
    x, y = np.array([[base_value, base_value], [0.13, 0.25]])
    line = lines.Line2D(x, y, lw=2., color='#F2F2F2')
    line.set_clip_on(False)
    ax.add_line(line)

    textout_val = plt.text(base_value, 0.33, 'base value',
                           fontsize=12, alpha=0.5,
                           horizontalalignment='center')
    textout_val.set_bbox(dict(facecolor='white', edgecolor='white'))


def plot_higher_lower_element(value, text):
    plt.text(value - text, 0.405, 'higher',
             fontsize=13, color='#FF0D57',
             horizontalalignment='right')

    plt.text(value + text, 0.405, 'lower',
             fontsize=13, color='#1E88E5',
             horizontalalignment='left')

    plt.text(value, 0.4, r'$\leftarrow$',
             fontsize=13, color='#1E88E5',
             horizontalalignment='center')

    plt.text(value, 0.425, r'$\rightarrow$',
             fontsize=13, color='#FF0D57',
             horizontalalignment='center')


def visualize(expected_values, values, X, feature_names=None, out_index=0, index=0):
    if feature_names is None:
        feature_names = ['Feature ' + str(i)
                         for i in range(values[0].shape[1])]

    expected_value = expected_values[out_index]
    if len(values[0].shape) == 1:
        shap_value = values[out_index]
        feature = X
    else:
        shap_value = values[out_index][index, :]
        feature = X[index]
    out_value = np.sum(shap_value) + expected_value

    features = {}
    for i in filter(lambda j: shap_value[j] != 0, range(len(shap_value))):
        features[i] = {
            "effect": shap_value[i],
            "value": feature[i]
        }

    data_config = {
        "out_names": "f[x]",
        "base_value": expected_value,
        "out_value": out_value,
        "link": 'identity',
        "feature_names": feature_names,
        "features": features
    }

    fig = plot_additive_plot(data_config, figsize=(
        20, 3), text_rotation=0, min_perc=0.05)

    return fig
