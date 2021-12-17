# Copyright 2020,2021 Sony Corporation.
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

import nnabla as nn
import nnabla.functions as F

import numpy as np


def get_cosine_annealing_learning_rate(curr_iter, T_max=500000, eta_max=5e-4, eta_min=1e-8):
    """
        cosine annealing scheduler.
    """
    lr = eta_min + 0.5 * (eta_max - eta_min) * \
        (1 + np.cos(np.pi*(curr_iter / T_max)))
    return lr


def get_direction_grid(height, width, focal_length, return_ij_2d_grid=False):
    """Forms a mesh grid for a given height and width and assumes the camera position to be fixed at the center of the the grid 
    (with a sufficiently large enough offset in z direction). Based on the prefixed camera position, 
    computes ray direction for every point in the grid.

    Args:
        height (int): Height of the image/grid
        width (int): Width of the image/grid
        focal_length (float): Camera focal length (calibrated intrinsics)

    Returns:
        directions (nn.Variable or nn.NdArray): Shape is (height, width, 3) - direction of projected ray for every grid point.
    """
    x = F.arange(0, width)
    y = F.arange(0, height)

    xx, yy = F.meshgrid(x, y)

    if return_ij_2d_grid:
        return F.stack(*list(F.meshgrid(x, y, ij_indexing=True)), axis=2)

    directions = F.stack((xx-width*0.5)/focal_length, -(yy-height*0.5) /
                         focal_length, F.constant(-1, xx.shape), axis=2)
    return directions


def get_ray_bundle(height, width, focal_length, cam2world_mat):
    """Computed direction and center of each ray from camera to each pixel coordinate in the image (1 ray per pixel)

    Args:
        height (int): Height of the image/grid
        width (int): Width of the image/grid
        focal_length (float): Camera focal length (calibrated intrinsics)
        cam2world_mat (nn.Variable or nn.NdArray): Transformation matrix from camera coordinate system to world coordinate system

    Returns:
        ray_directions (nn.Variable or nn.NdArray): Shape is (height, width, 3) - Direction of each projected ray from camera to grid point
        ray_origins (nn.Variable or nn.NdArray): Shape is (height, width, 3) - Center of each ray from camera to grid point
    """
    if cam2world_mat.ndim == 3:
        cam2world_mat = cam2world_mat[0, :, :]

    directions = get_direction_grid(height, width, focal_length)
    ray_directions = F.sum(
        directions[..., None, :]*cam2world_mat[None, None, :3, :3], axis=-1)
    ray_origins = F.broadcast(F.reshape(
        cam2world_mat[:3, -1], (1, 1, 3)), ray_directions.shape)

    return ray_directions, ray_origins


def compute_sample_points_for_variable_depth(ray_origins, ray_directions, near_plane, far_plane, num_samples, randomize=False):

    depth_steps = F.arange(0, 1 + 1/num_samples, 1/(num_samples-1))
    depth_steps = F.broadcast(
        depth_steps[None, :], (far_plane.shape[0], depth_steps.shape[0]))
    depth_values = near_plane[:, None] * \
        (1-depth_steps) + far_plane[:, None] * depth_steps

    if randomize:
        depth_vals_mid = 0.5 * (depth_values[:, :-1] + depth_values[:, 1:])
        # get intervals between samples
        upper = F.concatenate(depth_vals_mid, depth_values[:, -1:], axis=-1)
        lower = F.concatenate(depth_values[:, :1], depth_vals_mid, axis=-1)

        noise = F.rand(shape=depth_values.shape)
        depth_values = lower + (upper - lower) * noise

    sample_points = ray_origins[..., None, :] + \
        ray_directions[..., None, :]*depth_values[..., :, None]

    return sample_points, depth_values


def compute_sample_points_from_rays(ray_origins, ray_directions, near_plane, far_plane, num_samples, randomize=False):
    """Given a bundle of rays, this function samples points along each ray which is later used in volumetric rendering integration

    Args:
        ray_origins (nn.Variable or nn.NdArray): Shape is (height, width, 3) - Center of each ray from camera to grid point
        ray_directions (nn.Variable or nn.NdArray): Shape is (height, width, 3) - Direction of each projected ray from camera to grid point
        near_plane (float): Position of the near clipping plane
        far_plane (float): Position of the far clipping plane
        num_samples (int): Number of points to sample along each ray
        randomize (bool, optional): Defaults to True.

    Returns:
        sample_points: Shape is (height, width, num_samples, 3) - Sampled points along each ray
        depth_values: Shape is (num_samples, 1) - Depth values between the near and far plane at which point along each ray is sampled
    """

    if isinstance(near_plane, nn.Variable) or isinstance(near_plane, nn.NdArray):
        return compute_sample_points_for_variable_depth(ray_origins, ray_directions, near_plane, far_plane, num_samples, randomize)

    depth_values = F.arange(near_plane, far_plane+(far_plane-near_plane) /
                            num_samples, (far_plane-near_plane)/(num_samples-1))
    depth_values = F.reshape(
        depth_values, (1,)+depth_values.shape)
    if randomize:
        noise_shape = ray_origins.shape[:-1] + (num_samples,)
        if len(noise_shape) == 3:
            depth_values = depth_values[None, :, :] + F.rand(
                shape=noise_shape) * (far_plane-near_plane) / num_samples
        else:
            depth_values = depth_values + \
                F.rand(shape=noise_shape) * \
                (far_plane-near_plane) / num_samples

    sample_points = ray_origins[..., None, :] + \
        ray_directions[..., None, :]*depth_values[..., :, None]

    return sample_points, depth_values


def volumetric_rendering(radiance_field, ray_origins, depth_values, return_weights=False, white_bkgd=False, raw_noise_std=0.0, apply_act=False):
    """Integration of volumetric rendering

    Args:
        radiance_field (nn.Variable or nn.NdArray): Shape is (height, width, num_samples, 4). 
        radiance_field[:,:,:,:3] correspond to rgb value at each sampled point while radiance_field[:,:,:,-1] refers to color density.
        ray_origins (nn.Variable or nn.NdArray): Shape is (height, width, 3)
        depth_values (nn.Variable or nn.NdArray): Shape is (num_samples, 1) or (height, width, num_samples) 
        return_weights (bool, optional): Set to true if the coefficients of the volumetric integration sum are to be returned . Defaults to False.

    Returns:
        rgb_map (nn.Variable or nn.NdArray): Shape is (height, width, 3)
        rgb_map (nn.Variable or nn.NdArray): Shape is (height, width, 1)
    """
    if apply_act:
        sigma = F.relu(radiance_field[..., 3])
        rgb = F.sigmoid(radiance_field[..., :3])
    else:
        sigma = radiance_field[..., 3]
        rgb = radiance_field[..., :3]

    if raw_noise_std > 0.0:
        noise = F.randn(shape=sigma.shape)
        sigma += (noise*raw_noise_std)

    if depth_values.ndim == 2:
        distances = depth_values[:, 1:] - depth_values[:, :-1]
        distances = F.concatenate(distances, F.constant(
            1e2, shape=depth_values.shape[:-1]+(1,)), axis=-1)
        alpha = 1. - F.exp(-sigma*distances)
        weights = alpha * F.cumprod(1-alpha+1e-10, axis=-1, exclusive=True)
        rgb_map = F.sum(weights[..., None]*rgb, axis=-2)
        depth_map = F.sum(weights*depth_values, axis=-1)
        acc_map = F.sum(weights, axis=-1)
    else:
        distances = depth_values[:, :, 1:] - depth_values[:, :, :-1]
        distances = F.concatenate(distances, F.constant(
            1e10, shape=depth_values.shape[:-1]+(1,)), axis=-1)
        alpha = 1. - F.exp(-sigma*distances)
        rgb_map = F.sum(weights[..., None]*rgb, axis=rgb.ndim-2)
        depth_map = F.sum(weights*depth_values, axis=1)
        acc_map = F.sum(weights, axis=-1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[..., None])

    if return_weights:
        disp_map = 1.0 / \
            F.maximum2(F.constant(1e-10, depth_map.shape), depth_map / acc_map)
        return rgb_map, depth_map, acc_map, disp_map, weights

    return rgb_map, depth_map, acc_map


def volume_rendering_transient(radiance_field, ray_origins, depth_values,
                               return_weights=False, white_bkgd=False, raw_noise_std=0.0, beta_min=0.1):

    static_rgb = radiance_field[..., :3]
    static_sigma = radiance_field[..., 3]

    if radiance_field.shape[-1] > 4:
        transient_rgb = radiance_field[..., 4:7]
        transient_sigma = radiance_field[..., 7]
        transient_beta = radiance_field[..., 8]

    distances = depth_values[:, 1:] - depth_values[:, :-1]
    distances = F.concatenate(distances, F.constant(
        1e2, shape=depth_values.shape[:-1]+(1,)), axis=-1)

    static_alpha = 1. - F.exp(-static_sigma*distances)

    if radiance_field.shape[-1] > 4:
        transient_alpha = 1. - F.exp(-transient_sigma*distances)
        alpha = 1. - F.exp(-(static_sigma+transient_sigma)*distances)
        transmittance = F.cumprod(1-static_alpha+1e-10, axis=-1, exclusive=True) * \
            F.cumprod(1-transient_alpha+1e-10, axis=-1, exclusive=True)
    else:
        alpha = static_alpha
        transmittance = F.cumprod(
            1-static_alpha+1e-10, axis=-1, exclusive=True)

    # weights = alpha * F.cumprod(1-alpha+1e-10, axis=-1, exclusive=True)

    static_weights = static_alpha * transmittance
    if radiance_field.shape[-1] > 4:
        transient_weights = transient_alpha * transmittance
        weights = alpha * transmittance

    static_rgb_map = F.sum(static_weights[..., None]*static_rgb, axis=-2)

    if isinstance(radiance_field, nn.Variable) and radiance_field.shape[-1] > 4:
        transient_rgb_map = F.sum(
            transient_weights[..., None]*transient_rgb, axis=-2)
        rgb_map = static_rgb_map + transient_rgb_map
        beta = F.sum(transient_weights*transient_beta, axis=-1)
        beta += beta_min
        acc_map = F.sum(weights, axis=-1)
        if white_bkgd:
            rgb_map = rgb_map + (1.-acc_map[..., None])

    elif isinstance(radiance_field, nn.NdArray) and radiance_field.shape[-1] > 4:
        transient_rgb_map = F.sum(
            transient_weights[..., None]*transient_rgb, axis=-2)
        rgb_map = static_rgb_map + transient_rgb_map
        static_weights = static_alpha * \
            F.cumprod(1-static_alpha+1e-10, axis=-1, exclusive=True)
        static_rgb_map = F.sum(static_weights[..., None]*static_rgb, axis=-2)
        transient_weights = transient_alpha * \
            F.cumprod(1-transient_alpha+1e-10, axis=-1, exclusive=True)
        transient_rgb_map = F.sum(
            transient_weights[..., None]*transient_rgb, axis=-2)
        beta = F.sum(transient_weights*transient_beta, axis=-1) + beta_min
        # rgb_map = static_rgb_map + transient_rgb_map
        acc_map = F.sum(weights, axis=-1)
        if white_bkgd:
            rgb_map = rgb_map + (1.-acc_map[..., None])

    else:
        acc_map = F.sum(static_weights, axis=-1)

    # depth_map = F.sum(weights*depth_values, axis=-1)
    if white_bkgd:
        static_rgb_map = static_rgb_map + (1.-acc_map[..., None])

    if return_weights:
        return static_rgb_map, static_weights

    return rgb_map, weights, static_rgb_map, transient_rgb_map, beta


def sin_cos_positional_embedding(x, num_encoding_functions, include_input=True, log_sampling=True):
    """Given coordinate positions of sampling points as a (N,3) array, this functions returns embeds each point with the sine and cosine function

    Args:
        x (nn.Variable or nn.NdArray): Shape is (N, 3). 
        num_encoding_functions (int): number of frequencies to encode for each grid position
        include_input (bool, optional): Whether include the original grid position along with the encoding of the position. Defaults to True.
        log_sampling (bool, optional): Sample logarithmically and not linearly. Defaults to True.

    Returns:
        [nn.Variable or nn.NdArray]: (N, num_encoding_functions*3*2+3) if include_input is True else (N, num_encoding_functions*3*2)
    """

    encoding = [x] if include_input else []

    if log_sampling:
        frequency_increments = F.arange(0, num_encoding_functions)
        frequency_bands = F.pow2(F.constant(
            2, shape=frequency_increments.shape), frequency_increments)
    else:
        frequency_bands = F.arange(2**0, 2**(num_encoding_functions-1)+1e-5,
                                   (2**(num_encoding_functions-1)-1)/(num_encoding_functions-1.0))

    for freq in frequency_bands:
        for func in [F.sin, F.cos]:
            encoding.append(func(x * F.reshape(freq, (1, 1))))
    return F.concatenate(*encoding, axis=x.ndim-1)


def get_encoding_function(num_encoding_function, include_input=True, log_sampling=True):
    return lambda x: sin_cos_positional_embedding(x, num_encoding_function, include_input, log_sampling)


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    """Normalized device coordinate rays.

    Space such that the canvas is a cube with sides [-1, 1] in each axis.

    Args:
      H (int): Height in pixels.
      W (int): Width in pixels.
      focal (float):  Focal length of pinhole camera.
      near (float): Near depth bound for the scene.
      rays_o (nn.Variable or nn.NdArray): shape [batch_size, 3]. Camera origin.
      rays_d (nn.Variable or nn.NdArray): shape [batch_size, 3]. Ray direction.

    Returns:
      rays_o: array of shape [batch_size, 3]. Camera origin in NDC.
      rays_d: array of shape [batch_size, 3]. Ray direction in NDC.
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / (rays_d[..., 2]+1e-5)
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1./(H/(2.*focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1./(W/(2.*focal)) * \
        (rays_d[..., 0]/rays_d[..., 2] - rays_o[..., 0]/rays_o[..., 2])
    d1 = -1./(H/(2.*focal)) * \
        (rays_d[..., 1]/rays_d[..., 2] - rays_o[..., 1]/rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = F.stack(o0, o1, o2, axis=-1)
    rays_d = F.stack(d0, d1, d2, axis=-1)

    return rays_o, rays_d


def sample_pdf(bins, weights, N_samples, det=False):
    """Sample additional points for training fine network

    Args:
      bins: int. Height in pixels.
      weights: int. Width in pixels.
      N_samples: float. Focal length of pinhole camera.
      det

    Returns:
      samples: array of shape [batch_size, 3]. Depth samples for fine network
    """
    weights += 1e-5
    pdf = weights/F.sum(weights, axis=-1, keepdims=True)

    cdf = F.cumsum(pdf, axis=-1)
    # if isinstance(pdf, nn.Variable):
    #     cdf = nn.Variable.from_numpy_array(tf.math.cumsum(pdf.d, axis=-1))
    # else:
    #     cdf = nn.Variable.from_numpy_array(tf.math.cumsum(pdf.data, axis=-1)).data
    cdf = F.concatenate(F.constant(0, cdf[..., :1].shape), cdf, axis=-1)

    if det:
        u = F.arange(0., 1., 1/N_samples)
        u = F.broadcast(u[None, :], cdf.shape[:-1] + (N_samples,))
        u = u.data if isinstance(cdf, nn.NdArray) else u
    else:
        u = F.rand(shape=cdf.shape[:-1] + (N_samples,))

    indices = F.searchsorted(cdf, u, right=True)
    # if isinstance(cdf, nn.Variable):
    #     indices = nn.Variable.from_numpy_array(
    #         tf.searchsorted(cdf.d, u.d, side='right').numpy())
    # else:
    #     indices = nn.Variable.from_numpy_array(
    #         tf.searchsorted(cdf.data, u.data, side='right').numpy())
    below = F.maximum_scalar(indices-1, 0)
    above = F.minimum_scalar(indices, cdf.shape[-1]-1)
    indices_g = F.stack(below, above, axis=below.ndim)
    cdf_g = F.gather(cdf, indices_g, axis=-1,
                     batch_dims=len(indices_g.shape)-2)
    bins_g = F.gather(bins, indices_g, axis=-1,
                      batch_dims=len(indices_g.shape)-2)

    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    denom = F.where(F.less_scalar(denom, 1e-5),
                    F.constant(1, denom.shape), denom)
    t = (u-cdf_g[..., 0])/denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])

    return samples
