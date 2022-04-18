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
from neu.pointnet2_utils import set_abstraction_msg, set_abstraction


def pointnet2_classification_msg(point_cloud: nn.Variable, train: bool, num_classes: int = 40) -> nn.Variable:
    """Pointnet2 for classification proposed by Charles R. Qi et. al.

        See: https://arxiv.org/pdf/1706.02413.pdf

    Args:
        point_cloud (nn.Variable): point cloud, shape(batch, number of points, 3 or 6)
        train (bool): training flag
        num_classes (int): number of classes, default is 40

    Returns:
        nn.Variable: pred logits
    """
    batch_size, num_points, dim = point_cloud.shape
    point_feature = None if dim == 3 else point_cloud[:, :, 3:]

    with nn.parameter_scope("abstraction_msg1"):
        msg1_point_cloud, msg1_point_feature = set_abstraction_msg(
            point_cloud[:, :, :3],
            point_feature,
            num_new_point_cloud=512,
            radius_list=[0.1, 0.2, 0.4],
            num_samples_list=[16, 32, 128],
            conv_channels=[[32, 32, 64], [64, 64, 128], [64, 96, 128]],
            train=train,
        )

    with nn.parameter_scope("abstraction_msg2"):
        msg2_point_cloud, msg2_point_feature = set_abstraction_msg(
            msg1_point_cloud,
            msg1_point_feature,
            num_new_point_cloud=128,
            radius_list=[0.2, 0.4, 0.8],
            num_samples_list=[32, 64, 128],
            conv_channels=[[64, 64, 128], [128, 128, 256], [128, 128, 256]],
            train=train,
        )

    with nn.parameter_scope("abstraction_msg3"):
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

    point_feature = F.reshape(msg3_point_feature, (batch_size, 1024))

    with nn.parameter_scope("affine1"):
        affine_h1 = PF.affine(point_feature, 512, with_bias=False)
        affine_h1 = PF.batch_normalization(affine_h1, batch_stat=train)
        affine_h1 = F.relu(affine_h1)
        if train:
            affine_h1 = F.dropout(affine_h1, p=0.4)

    with nn.parameter_scope("affine2"):
        affine_h2 = PF.affine(affine_h1, 256, with_bias=False)
        affine_h2 = PF.batch_normalization(affine_h2, batch_stat=train)
        affine_h2 = F.relu(affine_h2)
        if train:
            affine_h2 = F.dropout(affine_h2, p=0.5)

    with nn.parameter_scope("affine3"):
        pred_logit = PF.affine(affine_h2, num_classes)

    return pred_logit


def pointnet2_classification_ssg(point_cloud: nn.Variable, train: bool, num_classes: int = 40) -> nn.Variable:
    """Pointnet2 for classification proposed by Charles R. Qi et. al.

        See: https://arxiv.org/pdf/1706.02413.pdf

    Args:
        point_cloud (nn.Variable): point cloud, shape(batch, number of points, 3 or 6)
        train (bool): training flag
        num_classes (int): number of classes, default is 40

    Returns:
        nn.Variable: pred logits
    """
    batch_size, num_points, dim = point_cloud.shape
    point_feature = None if dim == 3 else point_cloud[:, :, 3:]

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

    with nn.parameter_scope("abstraction_ssg3"):
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

    point_feature = F.reshape(ssg3_point_feature, (batch_size, 1024))

    with nn.parameter_scope("affine1"):
        affine_h1 = PF.affine(point_feature, 512, with_bias=False)
        affine_h1 = PF.batch_normalization(affine_h1, batch_stat=train)
        affine_h1 = F.relu(affine_h1)
        if train:
            affine_h1 = F.dropout(affine_h1, p=0.4)

    with nn.parameter_scope("affine2"):
        affine_h2 = PF.affine(affine_h1, 256, with_bias=False)
        affine_h2 = PF.batch_normalization(affine_h2, batch_stat=train)
        affine_h2 = F.relu(affine_h2)
        if train:
            affine_h2 = F.dropout(affine_h2, p=0.5)

    with nn.parameter_scope("affine3"):
        pred_logit = PF.affine(affine_h2, num_classes)

    return pred_logit
