# Copyright 2021 Sony Corporation.
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


def get_cosine_annealing_learning_rate(curr_iter, t_max, eta_max, eta_min=0):
    """
    cosine annealing scheduler.
    """
    learning_rate = eta_min + 0.5 * \
        (eta_max - eta_min) * (1 + np.cos(np.pi * (curr_iter / t_max)))
    return learning_rate


def get_repeated_cosine_annealing_learning_rate(current_iter, eta_max, eta_min, period,
                                                num_periods):
    """
    restart cosine learning rate function after every quarter of total iteration.
    """
    if current_iter >= period * num_periods:
        return eta_min

    mod_iter = current_iter % period
    current_lr = get_cosine_annealing_learning_rate(
        mod_iter, period, eta_max, eta_min)
    return current_lr


def get_multistep_learning_rate(current_iter, lr_steps, current_lr):
    if current_iter in lr_steps:
        current_lr *= 0.5
    return current_lr


if __name__ == "__main__":

    """
    Draw graph to see how cosine annealing rate affects the learning rate.
    This is for visualization purpose only - does not affect training
    """
    lr_l = []

    # for cosine annealing learnig rate scheduler
    train_size = 600000
    period = 150000
    num_periods = 4
    eta_max = 2e-4
    eta_min = 1e-7
    i = 0
    while (i < train_size):
        lr = get_repeated_cosine_annealing_learning_rate(
            i, eta_max, eta_min, period, num_periods)
        lr_l.append(lr)
        i += 1

    # to visualize the LR scheduler
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    import matplotlib.ticker as mtick

    mpl.style.use('default')
    import seaborn
    seaborn.set(style='whitegrid')
    seaborn.set_context('paper')

    plt.figure(1)
    plt.subplot(111)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.title('NNabla', fontsize=16, color='k')
    plt.plot(list(range(600000)), lr_l,
             linewidth=1.5, label='learning rate scheme')
    legend = plt.legend(loc='upper right', shadow=False)
    ax = plt.gca()
    labels = ax.get_xticks().tolist()
    for k, v in enumerate(labels):
        labels[k] = str(int(v / 1000)) + 'K'
    ax.set_xticklabels(labels)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))

    ax.set_ylabel('Learning rate')
    ax.set_xlabel('Iteration')
    fig = plt.gcf()
    plt.savefig('cosine-lr.png')
    print("completed")
