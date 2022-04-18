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

from typing import Optional, Tuple, Iterable
import numpy as np

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF


def compute_square_distance(source_point_cloud: nn.Variable, destination_point_cloud: nn.Variable) -> nn.Variable:
    """compute square distance between given point clouds

    Args:
        source_point_cloud (nn.Variable): shape (batch_size_source, num_points_source, dim_source)
        destination_point_cloud (nn.Variable): shape (batch_size_destination, num_points_destination, dim_destination)

    Returns:
        nn.Variable: distance from source point cloud to destination_point_cloud
            (batch_size, num_points_source, num_points_destination)
    """
    batch_size_source, num_points_source, dim_source = source_point_cloud.shape
    batch_size_destination, num_points_destination, dim_destination = destination_point_cloud.shape

    assert dim_source == 3
    assert dim_destination == 3
    assert batch_size_source == batch_size_destination

    # distance.shape = (batch_size, num_points_source, num_points_destination)
    source_squared = F.sum(
        source_point_cloud**2, axis=2).reshape((batch_size_source, num_points_source, 1))
    destination_squared = F.sum(destination_point_cloud**2, axis=2).reshape(
        (batch_size_destination, 1, num_points_destination)
    )
    both_squared = -2.0 * \
        F.batch_matmul(source_point_cloud, F.transpose(
            destination_point_cloud, (0, 2, 1)))

    return source_squared + destination_squared + both_squared


def farthest_point_sample(point_cloud: nn.Variable, num_samples: int) -> Tuple[nn.Variable, nn.Variable]:
    """sample farthest points from given point clouds

    Args:
        point_cloud (nn.Variable): shape (batch_size, num_points, 3)
        num_sample_points (int): number of samples

    Return:
        Tuple[nn.Variable, nn.Variable]: sample index and sample points, shape(batch_size, num_samples, 3)
    """
    batch_size, num_points, dim = point_cloud.shape
    assert dim == 3

    sample_points_idx = []
    # Initialize with huge value
    batch_distance = nn.Variable.from_numpy_array(
        np.ones((batch_size, num_points)) * 1e6)
    # Initialize with random value
    farthest_points_idx = F.randint(0, high=num_points, shape=(batch_size,))

    for _ in range(num_samples):
        sample_points_idx.append(farthest_points_idx.reshape((batch_size, 1)))

        # new_base_point.shape = (batch_size, 1, 3)
        new_base_point = index_batch_variable(
            point_cloud, farthest_points_idx.reshape((batch_size, 1)))

        distance = F.sum((point_cloud - new_base_point) ** 2, axis=2)

        # mask.shape = (batch_size, num_points)
        mask = F.less(distance, batch_distance)
        not_mask = F.logical_not(mask)
        batch_distance = batch_distance * not_mask + distance * mask

        farthest_points_idx = F.max(batch_distance, axis=1, only_index=True)

    # sample_points_idx.shape = (batch_size, num_samples)
    sample_points_idx = F.concatenate(*sample_points_idx, axis=1)
    sample_points_idx.need_grad = False

    sample_points = index_batch_variable(point_cloud, sample_points_idx)
    sample_points.need_grad = False
    return sample_points, sample_points_idx


def query_ball_point(
    radius: float, num_samples: int, point_cloud: nn.Variable, query_point_cloud: nn.Variable
) -> Tuple[nn.Variable, nn.Variable]:
    """compute ball point centered on query_point_cloud

    Args:
        radius (float): local region radius
        num_samples (int): max sample number in local region
        point_cloud (nn.Variable): shape (batch_size, num_points, 3)
        query_point_cloud (nn.Variable): shape (batch_size, num_query_points, 3)

    Return:
        Tuple[nn.Variable, nn.Variable]: sample_group_points shape (batch_size, num_queries, num_samples),
            sorted_group_indices, shape (batch_size, num_queries, num_samples)
    """
    batch_size, num_points, dim = point_cloud.shape
    assert dim == 3

    query_batch_size, num_queries, dim = query_point_cloud.shape
    assert dim == 3
    assert batch_size == query_batch_size
    assert num_samples <= num_points

    # group_indices.shape = (batch_size, num_queries, num_points)
    group_indices_array = np.tile(np.arange(num_points).reshape(
        (1, 1, num_points)), (batch_size, num_queries, 1))
    group_indices = nn.Variable.from_numpy_array(
        group_indices_array.astype(np.int32))

    # compute distance from query point cloud
    # square_distance.shape = (batch_size, num_queries, num_points)
    squared_distance = compute_square_distance(query_point_cloud, point_cloud)

    # if greater than radius**2, replace it with max index
    max_indices_array = np.ones(
        (batch_size, num_queries, num_points)) * num_points
    max_indices_var = nn.Variable.from_numpy_array(
        max_indices_array.astype(np.int32))
    greater_mask = F.greater_scalar(squared_distance, radius**2)
    greater_not_mask = F.logical_not(greater_mask)
    group_indices = group_indices * greater_not_mask + max_indices_var * greater_mask

    sorted_group_indices = F.sort(group_indices, axis=2)
    sorted_group_indices = sorted_group_indices[:, :, :num_samples]

    group_first_indices = F.tile(
        sorted_group_indices[:, :, 0].reshape(
            (batch_size, num_queries, 1)), (1, 1, num_samples)
    )

    max_index_mask = F.equal_scalar(sorted_group_indices, num_points)
    max_index_not_mask = F.logical_not(max_index_mask)
    sorted_group_indices = group_first_indices * \
        max_index_mask + sorted_group_indices * max_index_not_mask
    sorted_group_indices.need_grad = False

    # sample (batch_size, num_queries, num_samples, 3)
    sample_group_points = index_batch_variable(
        point_cloud, sorted_group_indices.reshape(
            (batch_size, num_queries * num_samples))
    )
    sample_group_points = F.reshape(
        sample_group_points, (batch_size, num_queries, num_samples, 3))
    sample_group_points.need_grad = False
    return sample_group_points, sorted_group_indices


def index_batch_variable(batch_variable, batch_indices) -> nn.Variable:
    """index batch variable according to given batch indices

    Args:
        point_cloud (nn.Variable): shape (batch_size, num_points, dim)
        indices (nn.Variable): shape (batch_size, num_samples)

    Returns:
        nn.Variable: shape (batch_size, num_samples, dim)
    """
    batch_size, num_points, dim = batch_variable.shape

    indice_batch_size, num_indices = batch_indices.shape
    assert batch_size == indice_batch_size

    arange_array = np.tile(np.arange(batch_size)[:, np.newaxis], (1, num_indices))[
                           np.newaxis, :, :]
    arange_indices = nn.Variable.from_numpy_array(arange_array)

    indices = F.concatenate(arange_indices, batch_indices.reshape(
        (1, batch_size, num_indices)), axis=0)
    # indexed_variable.shape = (batch_size, num_samples, dim)
    indexed_variable = F.gather_nd(batch_variable, indices)

    return indexed_variable


def sample_and_group_all(point_cloud: nn.Variable, point_feature: nn.Variable) -> Tuple[nn.Variable, nn.Variable]:
    """sample and group all points

    Args:
        point_cloud (nn.Variable): shape (batch_size, num_points, 3)
        point_feature (nn.Variable): shape (batch_size, num_points, feature_dim)

    Returns:
        Tuple[nn.Variable, nn.Variable]: new_point_cloud (all elements are zeros), new_point_feature

    """
    batch_size, num_points, dim = point_cloud.shape
    assert dim == 3

    feature_batch_size, num_features, feature_dim = point_feature.shape
    assert num_features == num_points
    assert batch_size == feature_batch_size

    new_point_cloud_array = np.zeros((batch_size, 1, dim))
    new_point_cloud = nn.Variable.from_numpy_array(new_point_cloud_array)

    grouped_point_cloud = F.reshape(
        point_cloud, (batch_size, 1, num_points, dim))

    if point_feature is not None:
        new_point_feature = F.concatenate(
            grouped_point_cloud, point_feature.reshape((batch_size, 1, num_points, -1)), axis=3
        )
    else:
        new_point_feature = grouped_point_cloud

    return new_point_cloud, new_point_feature


def sample_and_group(
    num_new_point_cloud: int,
    radius: float,
    num_samples: int,
    point_cloud: nn.Variable,
    point_feature: Optional[nn.Variable],
) -> Tuple[nn.Variable, nn.Variable]:
    """sample and group points

     Args:
        num_new_point_clouds (int): number of new points
        radius (float): radius for query ball point
        num_samples (int): num samples for query ball point
        point_cloud (nn.Variable): shape (batch_size, num_points, 3)
        point_feature (nn.Variable): shape (batch_size, num_points, feature_dim)

    Returns:
        Tuple[nn.Variable, nn.Variable]: new_point_cloud shape (batch_size, num_new_point_cloud, dim), \
            new_point_feature (batch_size, num_new_point_cloud, num_samples, feature_dim+dim)
    """
    batch_size, num_points, dim = point_cloud.shape
    assert dim == 3
    if point_feature is not None:
        feature_batch_size, num_features, feature_dim = point_feature.shape

    new_point_cloud, _ = farthest_point_sample(
        point_cloud, num_new_point_cloud)
    # grouped_point_cloud.shape = (batch_size, num_new_point_cloud, num_query_ball_samples, 3)
    grouped_point_cloud, grouped_idx = query_ball_point(
        radius, num_samples, point_cloud, new_point_cloud)
    grouped_point_cloud = grouped_point_cloud - \
        F.reshape(new_point_cloud, (batch_size, num_new_point_cloud, 1, dim))

    if point_feature is not None:
        # grouped_feature.shape = (batch_size, num_new_point_cloud*num_samples, feature_dim)
        grouped_feature = index_batch_variable(
            point_feature, grouped_idx.reshape((batch_size, -1)))
        # grouped_feature.shape = (batch_size, num_new_point_cloud, num_samples, feature_dim)
        grouped_feature = F.reshape(
            grouped_feature, (batch_size, num_new_point_cloud, num_samples, feature_dim))
        new_point_feature = F.concatenate(
            grouped_point_cloud, grouped_feature, axis=3)
    else:
        new_point_feature = grouped_point_cloud

    return new_point_cloud, new_point_feature


def set_abstraction_msg(
    point_cloud: nn.Variable,
    point_feature: Optional[nn.Variable],
    num_new_point_cloud: int,
    radius_list: Iterable[float],
    num_samples_list: Iterable[int],
    conv_channels: Iterable[Iterable[int]],
    train: bool,
) -> Tuple[nn.Variable, nn.Variable]:
    """set abstraction layer using multi scale grouping (msg).

    Args:
        point_cloud (nn.Variable): shape (batch_size, num_points, dim)
        point_feature (Union[nn.Variable, None]): shape (batch_size, num_new_point_cloud, feature_dim)
        num_new_point_cloud (int): number of new point cloud
        radius_list (Iterable[float]): list of radius
        num_samples_list (Iterable[int]): list of number of samples
        conv_channels (Iterable[Iterable[int]]): list of convolutional channel
        train (bool): training flag

    Returns:
        Tuple[nn.Variable, nn.Variable]: new_point_cloud, shape (batch_size, len(radius_list)*num_new_point_cloud, 3)
            new_point_features, shape (batch_size, len(radius_list)*num_new_point_cloud, feature_dim)
    """
    batch_size, num_points, dim = point_cloud.shape
    assert dim == 3

    if point_feature is not None:
        feature_batch_size, num_features, feature_dim = point_feature.shape
        assert feature_batch_size == batch_size

    # sample new point cloud
    new_point_cloud, _ = farthest_point_sample(
        point_cloud, num_new_point_cloud)
    new_point_features = []

    assert len(num_samples_list) == len(radius_list)
    for i, (num_samples, radius) in enumerate(zip(num_samples_list, radius_list)):
        # grouped_idx.shape = (batch_size, num_new_point_cloud, num_samples)
        # gropued_point_cloud.shape = (batch_size, num_new_point_cloud, num_samples, 3)
        grouped_point_cloud, grouped_idx = query_ball_point(
            radius, num_samples, point_cloud, new_point_cloud)
        grouped_point_cloud = grouped_point_cloud - F.reshape(
            new_point_cloud, (batch_size, num_new_point_cloud, 1, dim)
        )

        if point_feature is not None:
            # grouped_feature.shape = (batch_size, num_new_point_cloud*num_samples, feature_dim)
            grouped_feature = index_batch_variable(
                point_feature, grouped_idx.reshape((batch_size, -1)))
            # grouped_feature.shape = (batch_size, num_new_point_cloud, num_samples, feature_dim)
            grouped_feature = F.reshape(
                grouped_feature, (batch_size, num_new_point_cloud, num_samples, feature_dim))
            grouped_feature = F.concatenate(
                grouped_feature, grouped_point_cloud, axis=-1)
        else:
            grouped_feature = grouped_point_cloud

        # grouped_feature.shape = (batch_size, num_new_point_cloud, num_samples, feature_dim)
        # to (batch_size, feature_dim, num_samples, num_new_point_cloud)
        grouped_feature = F.transpose(grouped_feature, (0, 3, 2, 1))
        conv_h = grouped_feature

        for j, conv_channel in enumerate(conv_channels[i]):
            with nn.parameter_scope(f"abstraction{i}/conv{j}"):
                conv_h = PF.convolution(
                    conv_h,
                    conv_channel,
                    (1, 1),
                    stride=(1, 1),
                    with_bias=False,
                )
                conv_h = PF.batch_normalization(conv_h, batch_stat=train)
                conv_h = F.relu(conv_h)

        # take max axis of num_samples
        # to (batch_size, feature_dim, num_new_point_cloud)
        new_point_feature = F.max(conv_h, axis=2)
        new_point_features.append(new_point_feature)

    new_point_features = F.concatenate(*new_point_features, axis=1)
    # to (batch_size, len(radius_list)*num_new_point_cloud, feature_dim)
    new_point_features = F.transpose(new_point_features, (0, 2, 1))
    return new_point_cloud, new_point_features


def set_abstraction(
    point_cloud: nn.Variable,
    point_feature: Optional[nn.Variable],
    num_new_point_cloud: Optional[int],
    radius: Optional[float],
    num_samples: int,
    conv_channels: Iterable[int],
    group_all: bool,
    train: bool,
) -> Tuple[nn.Variable, nn.Variable]:
    """set abstraction layer

    Args:
        point_cloud (nn.Variable): shape (batch_size, num_points, dim)
        point_feature (Union[nn.Variable, None]): shape (batch_size, num_new_point_cloud, feature_dim)
        num_new_point_cloud (int): number of new point cloud
        radius (float): radius
        num_samples (int): number of samples
        conv_channels (Iterable[int]): list of convolutional channel
        group_all (bool): grouping the point
        train (bool): training flag

    Returns:
        Tuple[nn.Variable, nn.Variable]: new_point_cloud, shape (batch_size, num_new_point_cloud, 3),
            new_point_features, shape (batch_size, num_new_point_cloud, feature_dim)
    """
    if group_all:
        new_point_cloud, new_point_feature = sample_and_group_all(
            point_cloud, point_feature)
    else:
        new_point_cloud, new_point_feature = sample_and_group(
            num_new_point_cloud, radius, num_samples, point_cloud, point_feature
        )

    # new_point_feature.shape = (batch_size, num_new_point_cloud, num_samples, feature_dim)
    # to (batch_size, feature_dim, num_samples, num_new_point_cloud)
    new_point_feature = F.transpose(new_point_feature, (0, 3, 2, 1))
    conv_h = new_point_feature

    for i, conv_channel in enumerate(conv_channels):
        with nn.parameter_scope(f"conv{i}"):
            conv_h = PF.convolution(
                conv_h,
                conv_channel,
                (1, 1),
                stride=(1, 1),
                with_bias=False,
            )
            conv_h = PF.batch_normalization(conv_h, batch_stat=train)
            conv_h = F.relu(conv_h)

    # take max axis of num_samples
    # to (batch_size, feature_dim, num_new_point_cloud)
    new_point_feature = F.max(conv_h, axis=2)

    # to (batch_size, num_new_point_cloud, feature_dim)
    new_point_feature = F.transpose(new_point_feature, (0, 2, 1))
    return new_point_cloud, new_point_feature


def feature_propagation(
    point_cloud_base: nn.Variable,
    point_feature_base: nn.Variable,
    point_cloud_next: nn.Variable,
    point_feature_next: nn.Variable,
    conv_channels: Iterable[int],
    train: bool,
    k: int = 3,
) -> nn.Variable:
    """feature propagation layer

    Args:
        point_cloud_base (nn.Variable): shape (batch_size, num_base_points, dim)
        point_feature_base (nn.Variable): shape (batch_size, num_base_points, feature_dim)
        point_cloud_next (nn.Variable): shape (batch_size, num_next_points, dim)
        point_feature_next (nn.Variable): shape (batch_size, num_next_points, feature_dim)
        train (bool): training flag
        k (int): knn parameter. Defaults to 3.

    Returns:
        nn.Variable: new_features, shape (batch_size, num_new_features, feature_dim)
    """
    batch_size, num_base_points, base_dim = point_cloud_base.shape
    _, num_next_points, _ = point_cloud_next.shape

    if num_next_points == 1:
        interpolated_point_feature = F.tile(
            point_feature_next, (1, num_base_points, 1))
    else:
        # distance.shape(batch_size, num_base_points, num_next_points)
        distance = compute_square_distance(point_cloud_base, point_cloud_next)
        sorted_distance, sorted_distance_idx = F.sort(
            distance, axis=2, with_index=True)
        sorted_distance = sorted_distance[:, :, :k]
        sorted_distance_idx = sorted_distance_idx[:, :, :k]
        batch_size, num_base_points, _ = sorted_distance_idx.shape

        inv_distance_batch = 1.0 / (sorted_distance + 1e-8)
        norm_batch = F.sum(inv_distance_batch, axis=2, keepdims=True)
        weight = inv_distance_batch / norm_batch
        # weight.shape(batch_size, num_base_points, 3, 1)
        weight = F.reshape(weight, (batch_size, num_base_points, 3, 1))

        sorted_distance_idx = F.reshape(
            sorted_distance_idx, (batch_size, num_base_points * 3))
        # selected_point_feature_next.shape(batch_size, num_base_points*3, point_next_feature_dim)
        selected_point_feature_next = index_batch_variable(
            point_feature_next, sorted_distance_idx)
        # selected_point_feature_next.shape(batch_size, num_base_points, 3, point_next_feature_dim)
        selected_point_feature_next = F.reshape(
            selected_point_feature_next, (batch_size, num_base_points, 3, -1))

        # sum according to num_next_points axis
        # selected_point_feature_next.shape(batch_size, num_base_points, 3, point_next_feature_dim)
        interpolated_point_feature = F.sum(
            selected_point_feature_next * weight, axis=2)

    if point_feature_base is not None:
        new_point_feature = F.concatenate(
            point_feature_base, interpolated_point_feature, axis=2)
    else:
        new_point_feature = interpolated_point_feature

    # new_point_feature.shape = (batch_size, num_base_points, feature_dim+point_next_feature_dim)
    # to (batch_size, feature_dim+point_next_feature_dim, num_base_points)
    new_point_feature = F.transpose(new_point_feature, (0, 2, 1))
    # add dimensions, to (batch_size, feature_dim, 1, num_base_points)
    new_point_feature = F.reshape(
        new_point_feature, (batch_size, -1, 1, num_base_points))

    conv_h = new_point_feature
    for i, conv_channel in enumerate(conv_channels):
        with nn.parameter_scope(f"conv{i}"):
            conv_h = PF.convolution(
                conv_h,
                conv_channel,
                (1, 1),
                stride=(1, 1),
            )
            conv_h = PF.batch_normalization(conv_h, batch_stat=train)
            conv_h = F.relu(conv_h)

    new_point_feature = F.reshape(conv_h, (batch_size, -1, num_base_points))
    new_point_feature = F.transpose(new_point_feature, (0, 2, 1))
    return new_point_feature
