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

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF

from model_utils import point_cloud_transform_net, feature_transform_net


def pointnet_feature_extraction(point_cloud: nn.Variable, train: bool) -> Tuple[nn.Variable, Dict[str, nn.Variable]]:
    """pointnet feature extraction proposed by Charles R. Qi et. al.
        See: https://arxiv.org/pdf/1612.00593.pdf

    Args:
        point_cloud (nn.Variable): point cloud, shape(batch, number of points, 3)
        train (bool): training flag

    Returns:
        Tuple[nn.Variable, Dict[str, nn.Variable]]: pointnet feature and internal variables
    """
    batch_size, num_points, _ = point_cloud.shape

    with nn.parameter_scope("tnet1"):
        point_cloud_transformation_mat, _ = point_cloud_transform_net(
            point_cloud, train)

    transformed_point_cloud = F.batch_matmul(
        point_cloud, point_cloud_transformation_mat)
    # expand dim to B*C(=K)*H(=num_points)*W(=dim)
    input_point_cloud = F.reshape(
        transformed_point_cloud, (batch_size, 1, num_points, 3))

    with nn.parameter_scope("conv1"):
        conv_h1 = PF.convolution(
            input_point_cloud, 64, (1, 3), stride=(1, 1), with_bias=False)
        conv_h1 = PF.batch_normalization(conv_h1, batch_stat=train)
        conv_h1 = F.relu(conv_h1)
        conv_h1 = F.transpose(conv_h1, (0, 2, 3, 1))

    with nn.parameter_scope("tnet2"):
        feature_transformation_mat, _ = feature_transform_net(
            conv_h1, train, K=64)

    transformed_feature = F.batch_matmul(
        conv_h1[:, :, 0, :], feature_transformation_mat)
    # expand dim to B*H(=num_points)*W(=dim)*C(=K)
    input_feature = F.reshape(
        transformed_feature, (batch_size, num_points, 1, 64))
    # B*H(=num_points)*W(=dim)*C(=K) to B*C(=K)*H(=num_points)*W(=dim)
    input_feature = F.transpose(input_feature, (0, 3, 1, 2))

    with nn.parameter_scope("conv2"):
        conv_h2 = PF.convolution(
            input_feature, 128, (1, 1), stride=(1, 1), with_bias=False)
        conv_h2 = PF.batch_normalization(conv_h2, batch_stat=train)
        conv_h2 = F.relu(conv_h2)

    with nn.parameter_scope("conv3"):
        conv_h3 = PF.convolution(
            conv_h2, 1024, (1, 1), stride=(1, 1), with_bias=False)
        conv_h3 = PF.batch_normalization(conv_h3, batch_stat=train)
        conv_h3 = F.relu(conv_h3)

    pool_h = F.max_pooling(conv_h3, (num_points, 1))
    pool_h = F.reshape(pool_h, (batch_size, -1))

    return pool_h, {
        "transformed_point_cloud": transformed_point_cloud,
        "point_cloud_transformation_mat": point_cloud_transformation_mat,
        "conv_h1": conv_h1,
        "conv_h2": conv_h2,
        "feature_transformation_mat": feature_transformation_mat,
        "transformed_feature": transformed_feature,
        "conv_h3": conv_h3,
        "pool_h": pool_h,
    }


def pointnet_classification(
    point_cloud: nn.Variable, train: bool, num_classes: int = 40
) -> Tuple[nn.Variable, Dict[str, nn.Variable]]:
    """Pointnet for classification proposed by Charles R. Qi et. al.
        See: https://arxiv.org/pdf/1612.00593.pdf

    Args:
        point_cloud (nn.Variable): point cloud, shape(batch, number of points, 3)
        train (bool): training flag
        num_classes (int): number of classes, default is 40

    Returns:
        Tuple[nn.Variable, Dict[str, nn.Variable]]: pred logits and internal variables
    """
    with nn.parameter_scope("pointnet_feature"):
        point_feature, internal_variables = pointnet_feature_extraction(
            point_cloud, train=train)

    with nn.parameter_scope("affine1"):
        affine_h1 = PF.affine(point_feature, 512, with_bias=False)
        affine_h1 = PF.batch_normalization(affine_h1, batch_stat=train)
        affine_h1 = F.relu(affine_h1)

    with nn.parameter_scope("affine2"):
        affine_h2 = PF.affine(affine_h1, 256, with_bias=False)
        affine_h2 = PF.batch_normalization(affine_h2, batch_stat=train)
        affine_h2 = F.relu(affine_h2)
        if train:
            affine_h2 = F.dropout(affine_h2, p=0.2)

    with nn.parameter_scope("affine3"):
        pred_logit = PF.affine(affine_h2, num_classes)

    return pred_logit, {
        "pointnet_feature_internal_variables": internal_variables,
        "pointnet_feature": point_feature,
        "affine_h1": affine_h1,
        "affine_h2": affine_h2,
        "pred_logit": pred_logit,
    }
