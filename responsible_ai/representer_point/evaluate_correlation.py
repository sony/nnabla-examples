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

import matplotlib.pyplot as plt
import h5py
import numpy as np
import nnabla as nn
import nnabla.functions as F

from scipy.stats.stats import pearsonr

with h5py.File("./data/info/no_shuffle/info.h5", "r") as hf:
    train_feature = hf["feature"]["train"][:]
    test_feature = hf["feature"]["test"][:]
    train_output = hf["output"]["train"][:]
    test_output = hf["output"]["test"][:]
feature_dict = {"train": train_feature, "test": test_feature}
output_dict = {"train": train_output, "test": test_output}

weight_matrix = np.load("./data/info/no_shuffle/weight_matrix.npy")

predicted_dict = {}

for phase, feature in feature_dict.items():
    w = np.matmul(train_feature.T, weight_matrix)
    temp = np.matmul(feature, w)
    softmax = F.softmax(nn.Variable.from_numpy_array(temp))
    softmax.forward()
    predicted_dict[phase] = softmax.d


np.random.seed(401)
train_rand_idx = np.random.choice(50000, 2000, replace=False)
np.random.seed(4)
test_rand_idx = np.random.choice(10000, 1000, replace=False)
rand_idx_dict = {"train": train_rand_idx, "test": test_rand_idx}


fontsize = 13
fig, ax = plt.subplots(1, 2)
for ind, phase in enumerate(["train", "test"]):

    output = output_dict[phase]
    pred = predicted_dict[phase]
    print(pred.shape)
    print(output.shape)
    corr, _ = pearsonr(pred.reshape(-1), output.reshape(-1))
    ax[ind].scatter(
        pred[rand_idx_dict[phase]], output[rand_idx_dict[phase]], marker="."
    )
    ax[ind].set_title("Correlation: %f" % corr)
    ax[ind].set_xlabel(f"Predicted {phase} Output", fontsize=fontsize)
    ax[ind].set_ylabel(f"Actual {phase} Output", fontsize=fontsize)
    ax[ind].set_aspect("equal")

plt.tight_layout()
plt.savefig("./data/info/no_shuffle/correlation.png", dpi=200)
