# Copyright 2022 Sony Group Corporation.
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

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF

# Install neu (nnabla examples utils) to import these functions.
# See [NEU](https://github.com/nnabla/nnabla-examples/tree/master/utils).
from neu.pointnet2_utils import set_abstraction_msg, set_abstraction, feature_propagation


def pointnet2_part_segmentation_msg(
    point_cloud: nn.Variable,
    one_hot_class_label: nn.Variable,
    train: bool,
    num_parts: int = 50,
    num_classes: int = 16,
) -> nn.Variable:
    """Pointnet2 for part segmentation proposed by Charles R. Qi et. al.

        See: https://arxiv.org/pdf/1706.02413.pdf

    Args:
        point_cloud (nn.Variable): point cloud, shape(batch, number of points, 3 or 6)
        one_hot_class_label (nn.Variable): one hot class label, shape (batch_size, num_classes)
        train (bool): training flag
        num_parts (int): number of parts, defaults to 50
        num_classes (int): number of classes, defaults to 16

    Returns:
        nn.Variable: pred logits, shape (batch_size, num_points, num_parts)
    """
    batch_size, num_points, dim = point_cloud.shape
    # If without normal, use point as point feature
    point_feature = point_cloud if dim == 3 else point_cloud[:, :, 3:]

    with nn.parameter_scope("abstraction_msg1"):
        msg1_point_cloud, msg1_point_feature = set_abstraction_msg(
            point_cloud[:, :, :3],
            point_feature,
            num_new_point_cloud=512,
            radius_list=[0.1, 0.2, 0.4],
            num_samples_list=[32, 64, 128],
            conv_channels=[[32, 32, 64], [64, 64, 128], [64, 96, 128]],
            train=train,
        )

    with nn.parameter_scope("abstraction_msg2"):
        msg2_point_cloud, msg2_point_feature = set_abstraction_msg(
            msg1_point_cloud,
            msg1_point_feature,
            num_new_point_cloud=128,
            radius_list=[0.4, 0.8],
            num_samples_list=[64, 128],
            conv_channels=[[128, 128, 256], [128, 196, 256]],
            train=train,
        )

    with nn.parameter_scope("abstraction"):
        msg3_point_cloud, msg3_point_feature = set_abstraction(
            msg2_point_cloud,
            msg2_point_feature,
            num_new_point_cloud=None,
            radius=None,
            num_samples=None,
            conv_channels=[256, 512, 1024],
            group_all=True,
            train=train,
        )

    with nn.parameter_scope("feature_propagation1"):
        msg2_point_feature = feature_propagation(
            msg2_point_cloud,
            msg2_point_feature,
            msg3_point_cloud,
            msg3_point_feature,
            conv_channels=[256, 256],
            train=train,
        )

    with nn.parameter_scope("feature_propagation2"):
        msg1_point_feature = feature_propagation(
            msg1_point_cloud,
            msg1_point_feature,
            msg2_point_cloud,
            msg2_point_feature,
            conv_channels=[256, 128],
            train=train,
        )

    one_hot_class_label = F.tile(F.reshape(
        one_hot_class_label, (batch_size, num_classes, 1)), (1, 1, num_points))
    one_hot_class_label = F.transpose(one_hot_class_label, (0, 2, 1))
    # point_feature.shape = (batch_size, num_poins, 16+3+3)
    point_feature = F.concatenate(
        one_hot_class_label, point_cloud[:, :, :3], point_feature, axis=2)

    with nn.parameter_scope("feature_propagation3"):
        point_feature = feature_propagation(
            point_cloud[:, :, :3],
            point_feature,
            msg1_point_cloud,
            msg1_point_feature,
            conv_channels=[128, 128, 128],
            train=train,
        )

    # point_feature.shape = (batch_size, num_points, num_features)
    batch_size, new_num_points, new_num_features = point_feature.shape
    assert new_num_points == num_points

    point_feature = F.transpose(point_feature, (0, 2, 1))
    point_feature = F.reshape(
        point_feature, (batch_size, new_num_features, 1, new_num_points))

    with nn.parameter_scope("conv1"):
        conv_h1 = PF.convolution(point_feature, 128, [1, 1], with_bias=False)
        conv_h1 = PF.batch_normalization(conv_h1, batch_stat=train)
        conv_h1 = F.relu(conv_h1)
        if train:
            conv_h1 = F.dropout(conv_h1, p=0.5)

    with nn.parameter_scope("conv2"):
        conv_h2 = PF.convolution(conv_h1, num_parts, [1, 1])

    # pred_logit.shape = (batch_size, num_parts, num_points)
    pred_logits = conv_h2[:, :, 0, :]
    # pred_logit.shape = (batch_size, num_points, num_parts)
    pred_logits = F.transpose(pred_logits, (0, 2, 1))
    return pred_logits


def pointnet2_part_segmentation_ssg(
    point_cloud: nn.Variable,
    one_hot_class_label: nn.Variable,
    train: bool,
    num_parts: int = 50,
    num_classes: int = 16,
) -> nn.Variable:
    """Pointnet2 for part segmentation proposed by Charles R. Qi et. al.

        See: https://arxiv.org/pdf/1706.02413.pdf

    Args:
        point_cloud (nn.Variable): point cloud, shape(batch, number of points, 3 or 6)
        one_hot_class_label (nn.Variable): one hot class label, shape (batch_size, num_classes)
        train (bool): training flag
        num_parts (int): number of parts, defaults to 50
        num_classes (int): number of classes, defaults to 16

    Returns:
        nn.Variable: pred logits, shape (batch_size, num_points, num_parts)
    """
    batch_size, num_points, dim = point_cloud.shape
    # If without normal, use point as point feature
    point_feature = point_cloud if dim == 3 else point_cloud[:, :, 3:]

    with nn.parameter_scope("abstraction_ssg1"):
        ssg1_point_cloud, ssg1_point_feature = set_abstraction(
            point_cloud[:, :, :3],
            point_feature,
            num_new_point_cloud=512,
            radius=0.2,
            num_samples=32,
            conv_channels=[64, 64, 128],
            group_all=False,
            train=train,
        )

    with nn.parameter_scope("abstraction_ssg2"):
        ssg2_point_cloud, ssg2_point_feature = set_abstraction(
            ssg1_point_cloud,
            ssg1_point_feature,
            num_new_point_cloud=128,
            radius=0.4,
            num_samples=64,
            conv_channels=[128, 128, 256],
            group_all=False,
            train=train,
        )

    with nn.parameter_scope("abstraction"):
        ssg3_point_cloud, ssg3_point_feature = set_abstraction(
            ssg2_point_cloud,
            ssg2_point_feature,
            num_new_point_cloud=None,
            radius=None,
            num_samples=None,
            conv_channels=[256, 512, 1024],
            group_all=True,
            train=train,
        )

    with nn.parameter_scope("feature_propagation1"):
        ssg2_point_feature = feature_propagation(
            ssg2_point_cloud,
            ssg2_point_feature,
            ssg3_point_cloud,
            ssg3_point_feature,
            conv_channels=[256, 256],
            train=train,
        )

    with nn.parameter_scope("feature_propagation2"):
        ssg1_point_feature = feature_propagation(
            ssg1_point_cloud,
            ssg1_point_feature,
            ssg2_point_cloud,
            ssg2_point_feature,
            conv_channels=[256, 128],
            train=train,
        )

    one_hot_class_label = F.tile(F.reshape(
        one_hot_class_label, (batch_size, num_classes, 1)), (1, 1, num_points))
    one_hot_class_label = F.transpose(one_hot_class_label, (0, 2, 1))
    # point_feature.shape = (batch_size, num_poins, 16+3+3)
    point_feature = F.concatenate(
        one_hot_class_label, point_cloud[:, :, :3], point_feature, axis=2)

    with nn.parameter_scope("feature_propagation3"):
        point_feature = feature_propagation(
            point_cloud[:, :, :3],
            point_feature,
            ssg1_point_cloud,
            ssg1_point_feature,
            conv_channels=[128, 128, 128],
            train=train,
        )

    # point_feature.shape = (batch_size, num_points, num_features)
    batch_size, new_num_points, new_num_features = point_feature.shape
    assert new_num_points == num_points

    point_feature = F.transpose(point_feature, (0, 2, 1))
    point_feature = F.reshape(
        point_feature, (batch_size, new_num_features, 1, new_num_points))

    with nn.parameter_scope("conv1"):
        conv_h1 = PF.convolution(point_feature, 128, [1, 1], with_bias=False)
        conv_h1 = PF.batch_normalization(conv_h1, batch_stat=train)
        conv_h1 = F.relu(conv_h1)
        if train:
            conv_h1 = F.dropout(conv_h1, p=0.5)

    with nn.parameter_scope("conv2"):
        conv_h2 = PF.convolution(conv_h1, num_parts, [1, 1])

    # pred_logit.shape = (batch_size, num_parts, num_points)
    pred_logits = conv_h2[:, :, 0, :]
    # pred_logit.shape = (batch_size, num_points, num_parts)
    pred_logits = F.transpose(pred_logits, (0, 2, 1))
    return pred_logits
