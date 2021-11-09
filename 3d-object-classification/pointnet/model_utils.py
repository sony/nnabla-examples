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

from typing import Dict, Tuple
import numpy as np

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF


def point_cloud_transform_net(point_cloud: nn.Variable, train: bool) -> Tuple[nn.Variable, Dict[str, nn.Variable]]:
    """T net, create transformation matrix for point cloud

    Args:
        point_cloud (nn.Variable): point cloud, shape(batch, number of points, 3)
        train (bool): training flag

    Returns:
        Tuple[nn.Variable, Dict[str, nn.Variable]]: transformation matrix and internal variables
    """
    batch_size, num_points, _ = point_cloud.shape
    # expand dim to B*C(=K)*H(=num_points)*W(=dim)
    point_cloud = F.reshape(point_cloud, shape=(batch_size, 1, num_points, 3))

    with nn.parameter_scope("conv1"):
        conv_h1 = PF.convolution(
            point_cloud, 64, (1, 3), stride=(1, 1), with_bias=False)
        conv_h1 = PF.batch_normalization(conv_h1, batch_stat=train)
        conv_h1 = F.relu(conv_h1)

    with nn.parameter_scope("conv2"):
        conv_h2 = PF.convolution(conv_h1, 128, (1, 1),
                                 stride=(1, 1), with_bias=False)
        conv_h2 = PF.batch_normalization(conv_h2, batch_stat=train)
        conv_h2 = F.relu(conv_h2)

    with nn.parameter_scope("conv3"):
        conv_h3 = PF.convolution(
            conv_h2, 1024, (1, 1), stride=(1, 1), with_bias=False)
        conv_h3 = PF.batch_normalization(conv_h3, batch_stat=train)
        conv_h3 = F.relu(conv_h3)

    pool_h = F.max_pooling(conv_h3, (num_points, 1))
    pool_h = F.reshape(pool_h, (batch_size, -1))

    with nn.parameter_scope("affine1"):
        affine_h1 = PF.affine(pool_h, 512, with_bias=False)
        affine_h1 = PF.batch_normalization(affine_h1, batch_stat=train)
        affine_h1 = F.relu(affine_h1)

    with nn.parameter_scope("affine2"):
        affine_h2 = PF.affine(affine_h1, 256, with_bias=False)
        affine_h2 = PF.batch_normalization(affine_h2, batch_stat=train)
        affine_h2 = F.relu(affine_h2)

    with nn.parameter_scope("affine3"):
        # transform points (3 dim) so the matrix size is (3*3)
        transform_h = PF.affine(affine_h2, 3 * 3)
        eye_mat = nn.Variable.from_numpy_array(
            np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float32))
        eye_mat = F.reshape(eye_mat, (1, 9))
        transform_h = transform_h + eye_mat

    transform_h = F.reshape(transform_h, (batch_size, 3, 3))
    return transform_h, {
        "conv_h1": conv_h1,
        "conv_h2": conv_h2,
        "conv_h3": conv_h3,
        "pool_h": pool_h,
        "affine_h1": affine_h1,
        "affine_h2": affine_h2,
        "transform_h": transform_h,
    }


def feature_transform_net(feature: nn.Variable, train: bool, K: int = 64) -> Tuple[nn.Variable, Dict[str, nn.Variable]]:
    """T net, create transformation matrix

    Args:
        feature (nn.Variable): feature, shape(batch, number of points, 1, K)
        train (bool): training flag
        K (int): transformation matrix size, default is 64.

    Returns:
        Tuple[nn.Variable, Dict[str, nn.Variable]]: transformation matrix and internal variables
    """
    batch_size, num_points, *_ = feature.shape
    # B*H(=num_points)*W(=dim)*C(=K) to B*C(=K)*H(=num_points)*W(=dim)
    feature = F.transpose(feature, (0, 3, 1, 2))
    with nn.parameter_scope("conv1"):
        conv_h1 = PF.convolution(
            feature, 64, (1, 1), stride=(1, 1), with_bias=False)
        conv_h1 = PF.batch_normalization(conv_h1, batch_stat=train)
        conv_h1 = F.relu(conv_h1)

    with nn.parameter_scope("conv2"):
        conv_h2 = PF.convolution(conv_h1, 128, (1, 1),
                                 stride=(1, 1), with_bias=False)
        conv_h2 = PF.batch_normalization(conv_h2, batch_stat=train)
        conv_h2 = F.relu(conv_h2)

    with nn.parameter_scope("conv3"):
        conv_h3 = PF.convolution(
            conv_h2, 1024, (1, 1), stride=(1, 1), with_bias=False)
        conv_h3 = PF.batch_normalization(conv_h3, batch_stat=train)
        conv_h3 = F.relu(conv_h3)

    pool_h = F.max_pooling(conv_h3, (num_points, 1))
    pool_h = F.reshape(pool_h, (batch_size, -1))

    with nn.parameter_scope("affine1"):
        affine_h1 = PF.affine(pool_h, 512, with_bias=False)
        affine_h1 = PF.batch_normalization(affine_h1, batch_stat=train)
        affine_h1 = F.relu(affine_h1)

    with nn.parameter_scope("affine2"):
        affine_h2 = PF.affine(affine_h1, 256, with_bias=False)
        affine_h2 = PF.batch_normalization(affine_h2, batch_stat=train)
        affine_h2 = F.relu(affine_h2)

    with nn.parameter_scope("affine3"):
        transform_h = PF.affine(affine_h2, K * K)
        eye_mat = nn.Variable.from_numpy_array(
            np.eye(K, dtype=np.float32).flatten())
        eye_mat = F.reshape(eye_mat, (1, K * K))
        transform_h = transform_h + eye_mat

    transform_h = F.reshape(transform_h, (batch_size, K, K))
    return transform_h, {
        "conv_h1": conv_h1,
        "conv_h2": conv_h2,
        "conv_h3": conv_h3,
        "pool_h": pool_h,
        "affine_h1": affine_h1,
        "affine_h2": affine_h2,
        "transform_h": transform_h,
    }
