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
import nnabla.parametric_functions as PF
import nnabla.initializer as I
import numpy as np
from nnabla.ext_utils import get_extension_context
from nnabla.random import prng
from nnabla.parametric_functions import parametric_function_api
from nnabla.parameter import get_parameter_or_create

import matplotlib.pyplot as plt
from functools import partial

from ray_tracer import ray_trace


class WeightNormalizationInitializer(I.BaseInitializer):

    def __init__(self, w, dim=0, eps=1e-12):
        self.w = w
        self.dim = dim
        self.eps = eps

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__)

    def __call__(self, shape):
        axis = tuple([a for a in range(len(self.w.shape)) if a != self.dim])
        w_norm_data = np.sqrt(np.sum(self.w.d ** 2, axis=axis) + self.eps)
        return np.float32(w_norm_data)


@parametric_function_api("wn", [
    ('g', 'Weight Normalization adaptive scale scalar.', 'w.shape[dim]', True),
])
def weight_normalization(w, dim=1, eps=1e-12, fix_parameters=False):
    """
    """
    outmaps = w.shape[dim]
    initializer = WeightNormalizationInitializer(w, dim, eps)
    g = get_parameter_or_create("g", (outmaps, ),
                                initializer=initializer, need_grad=True,
                                as_need_grad=not fix_parameters)
    w_wn = F.weight_normalization(w, g, dim, eps=eps)
    return w_wn


def weight_normalization_backward(inputs, dim=1, eps=1e-12):
    # We do not need the gradients wrt weight and scale by nn.grad,
    # since nn.grad is performed wrt the intersection point.
    ## dw_wn = inputs[0]
    ## w0 = inputs[1]
    ## g0 = inputs[2]
    return None, None


nn.backward_functions.register(
    "WeightNormalization", weight_normalization_backward)


def affine(h, D, w_init=None, b_init=None, with_bias=True, name=None):
    with nn.parameter_scope(name):
        apply_w = partial(weight_normalization, dim=1)
        h = PF.affine(h, D, base_axis=h.ndim - 1, apply_w=apply_w,
                      w_init=w_init, b_init=b_init, with_bias=with_bias)
    return h


def positional_encoding(x, N=6, include_input=True):
    """
    Args:
      x: Input (B, R, 3)
      N: Number of bands, N=6 for implicit network and N=4 for rendering network.
    """

    gamma = [x] if include_input else []
    bands = 2 ** np.arange(0, N + 1)
    data_holder = nn.Variable if isinstance(x, nn.Variable) else nn.NdArray
    bands = data_holder.from_numpy_array(bands)
    bands = F.reshape(bands, tuple([1] * x.ndim) + (N + 1, )) \
        * F.reshape(x, x.shape + (1, ))
    bands = F.reshape(bands, bands.shape[:-2] + (-1, ))
    cos_x = F.cos(bands)
    sin_x = F.sin(bands)

    gamma += [cos_x, sin_x]
    gamma = F.concatenate(*gamma, axis=-1)

    return gamma


class GeometricInitializer(I.BaseInitializer):
    """
    """

    def __init__(self, Di, Do, sigma, zero_start=None):
        self.Di = Di
        self.Do = Do
        self.sigma = sigma
        self.zero_start = zero_start

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__)

    def __call__(self, shape):
        w_init = np.sqrt(self.sigma) * prng.randn(*[self.Di, self.Do])
        if self.zero_start is not None:
            w_init[self.zero_start:, :] = 0.0
        return w_init


@nn.parameter_scope("implicit-network")
def implicit_network(x, D=512, feature_size=256, L=9, skip_in=[4], N=6,
                     act="softplus",
                     including_input=True,
                     initial_sphere_radius=0.75):
    """Implicit Network.

    Args:
      x: Position on a ray.
      D: Dimension of a network.
      feature_size: Feature dimension of the final output.
      L: Number of layers
      skip_in: Where the skip connection appears.
      N: Number of frequency of the positional encoding.
      act: Activation function.
      inclugin_input: Include input to the positional encoding (PE).
      initial_sphere_radius: the radius of the initial network sphere.

    Network architecture looks like:

    x --> [PE(x)] --> affine --> relu --> ... -->
    concate([h, x]) --> affine --> relu --> ... -->
    affine(h) --> [sdf, feature]
    """

    act_map = dict(relu=F.relu, softplus=partial(F.softplus, beta=100))
    Dx = x.shape[-1]
    act = act_map[act]

    h = positional_encoding(x, N, including_input)
    for l in range(L):
        # First
        if l == 0:
            Dh = h.shape[-1]
            Dx = x.shape[-1]
            w_init = GeometricInitializer(Dh, D, 2 / D, Dx)
            h = affine(h, D, w_init=w_init, name=f"affine-{l:02d}")
            h = act(h)
        # Skip
        elif l in skip_in:
            w_init = GeometricInitializer(D, D, 2 / (D - Dx), -Dx)
            h = affine(h, D, w_init=w_init, name=f"affine-{l:02d}")
            h = act(h)
        # Last (scalar + feature_size)
        elif l == L - 1:
            Do = 1 + feature_size
            w_init = np.sqrt(np.pi / D) * np.ones([D, Do])
            h = affine(h, Do, w_init=w_init, b_init=I.ConstantInitializer(-initial_sphere_radius),
                       name=f"affine-last")
        # Intermediate
        else:
            Do = D - Dx if l + 1 in skip_in else D
            w_init = GeometricInitializer(D, Do, 2 / Do)
            h = affine(h, Do, w_init=w_init, name=f"affine-{l:02d}")
            h = act(h)
            h = F.concatenate(*[h, x]) if l + 1 in skip_in else h
            # h = F.concatenate(*[h, x]) / np.sqrt(2) if l + 1 in skip_in else h # (the paper used this scale)
    return h


def sdf_feature_grad(implicit_network, x, conf):
    y = implicit_network(x, initial_sphere_radius=conf.initial_sphere_radius)
    sdf = y[..., 0:1]
    feature = y[..., 1:]
    grad = nn.grad([sdf], [x])[0]
    return sdf, feature, grad


def sample_network(x_curr, sdf_cur, raydir, grad_curr):
    """
    x_curr: Points (B, R, 3) either on surface or not
    sdf_cur: SDF on x_curr (B, R, 1)
    raydir: Ray direction (B, R, 3)
    grad_curr: Gradients on x_curr (B, R, 3)
    """

    # Denominator
    de = F.batch_matmul(grad_curr[..., np.newaxis, :],
                        raydir[..., np.newaxis, :], transpose_b=True)
    de = de.reshape(sdf_cur.shape)
    de_inv = (1.0 / de).apply(need_grad=False)
    de_inv = F.minimum_scalar(de_inv, 1e30).apply(
        need_grad=False)  # (numerical issue de = cos(x, y) = 0)
    # Differentiable intersection point (discrete update of implicit differentiation)
    sdf_cur0 = sdf_cur.get_unlinked_variable(need_grad=False)
    x_hat = x_curr - (sdf_cur - sdf_cur0) * de_inv * raydir
    return x_hat


@nn.parameter_scope("lighting-network")
def lighting_network(x_hat, normal, feature, view, D=512, L=4, N=4, including_input=True):
    """
    Args
      x_hat: Differentiable intersection point (B, R, 3)
      normal: Normal on x_hat (B, R, 3) (should be normalized before).
      feature: Intermediate output of the implicit network (B, R, feature_size).
      view: View direction (B, R, 3)
      D: Dimension of a network.
      L: Number of layers.
      N: Number of frequency of the positional encoding.
      inclugin_input: Include input to the positional encoding (PE).
    """
    pe_view = positional_encoding(view, N, including_input)
    h = F.concatenate(*[x_hat, normal, feature, pe_view], axis=-1)
    for l in range(L - 1):
        h = affine(h, D, name=f"affine-{l:02d}")
        h = F.relu(h)
    h = affine(h, 3, name=f"affine-{L - 1:02d}")
    h = F.tanh(h)
    return h


def sdf_net(x, conf):
    out = implicit_network(x, D=conf.depth, feature_size=conf.feature_size,
                           initial_sphere_radius=conf.initial_sphere_radius)
    sdf = out[..., 0:1]
    return sdf


def norm(x, axis, eps=1e-24):
    return F.sum(x ** 2 + eps, axis, keepdims=True) ** 0.5


def idr_loss(camloc, raydir, alpha, color_gt, mask_obj, conf):
    # Setting
    B, R, _ = raydir.shape
    L = conf.layers
    D = conf.depth
    feature_size = conf.feature_size

    # Ray trace (visibility)
    x_hit, mask_hit, dists, mask_pin, mask_pout = \
        ray_trace(partial(sdf_net, conf=conf),
                  camloc, raydir, mask_obj, t_near=conf.t_near, t_far=conf.t_far,
                  sphere_trace_itr=conf.sphere_trace_itr,
                  ray_march_points=conf.ray_march_points,
                  n_chunks=conf.n_chunks,
                  max_post_itr=conf.max_post_itr,
                  post_method=conf.post_method, eps=conf.eps)

    x_hit = x_hit.apply(need_grad=False)
    mask_hit = mask_hit.apply(need_grad=False, persistent=True)
    dists = dists.apply(need_grad=False)
    mask_pin = mask_pin.apply(need_grad=False)
    mask_pout = mask_pout.apply(need_grad=False)
    mask_us = mask_pin + mask_pout
    P = F.sum(mask_us)

    # Current points
    x_curr = (camloc.reshape((B, 1, 3)) + dists * raydir).apply(need_grad=True)

    # Eikonal loss
    bounding_box_size = conf.bounding_box_size
    x_free = F.rand(-bounding_box_size, bounding_box_size,
                    shape=(B, R // 2, 3))
    x_point = F.concatenate(*[x_curr, x_free], axis=1)
    sdf_xp, _, grad_xp = sdf_feature_grad(implicit_network, x_point, conf)
    gp = (F.norm(grad_xp, axis=[grad_xp.ndim - 1], keepdims=True) - 1.0) ** 2.0
    loss_eikonal = F.sum(gp[:, :R, :] * mask_us) + F.sum(gp[:, R:, :])
    loss_eikonal = loss_eikonal / (P + B * R // 2)
    loss_eikonal = loss_eikonal.apply(persistent=True)

    sdf_curr = sdf_xp[:, :R, :]
    grad_curr = grad_xp[:, :R, :]

    # Mask loss
    logit = - alpha.reshape([1 for _ in range(sdf_curr.ndim)]) * sdf_curr
    loss_mask = F.sigmoid_cross_entropy(logit, mask_obj)
    loss_mask = loss_mask * mask_pout
    loss_mask = F.sum(loss_mask) / P / alpha
    loss_mask = loss_mask.apply(persistent=True)

    # Lighting
    x_hat = sample_network(x_curr, sdf_curr, raydir, grad_curr)
    _, feature, grad = sdf_feature_grad(implicit_network, x_hat, conf)
    normal = grad
    color_pred = lighting_network(x_hat, normal, feature, -raydir, D)

    # Color loss
    loss_color = F.absolute_error(color_gt, color_pred)
    loss_color = loss_color * mask_pin
    loss_color = F.sum(loss_color) / P
    loss_color = loss_color.apply(persistent=True)

    # Total loss
    loss = loss_color + conf.mask_weight * \
        loss_mask + conf.eikonal_weight * loss_eikonal

    return loss, loss_color, loss_mask, loss_eikonal, mask_hit


def render(camloc, raydir, conf):
    # Setting
    B, R, _ = raydir.shape
    L = conf.layers
    D = conf.depth
    feature_size = conf.feature_size

    # Sphere trace
    x_hit, mask_hit, _, _, _ = \
        ray_trace(partial(sdf_net, conf=conf),
                  camloc, raydir, test=True,
                  t_near=conf.t_near, t_far=conf.t_far,
                  sphere_trace_itr=conf.sphere_trace_itr,
                  ray_march_points=conf.ray_march_points,
                  n_chunks=conf.n_chunks,
                  max_post_itr=conf.max_post_itr,
                  post_method=conf.post_method, eps=conf.eps)
    x_hit = x_hit.apply(need_grad=True)
    mask_hit = mask_hit.apply(need_grad=False)

    # Intersection points and current points
    sdf_hit, feature, grad_xhit = sdf_feature_grad(
        implicit_network, x_hit, conf)

    # Lighting
    normal = grad_xhit
    raydir = raydir.reshape((B, R, 3))
    color_pred = lighting_network(x_hit, normal, feature, -raydir, D)
    color_pred = color_pred * mask_hit
    color_pred = color_pred.reshape((B, R, 3))

    return color_pred


def main(args):
    ctx = get_extension_context('cudnn', device_id=args.device_id)
    nn.set_default_context(ctx)

    grid_size = 50
    L = args.layers
    initial_sphere_radius = args.initial_sphere_radius

    mae_list = []
    for d in args.depth_list:
        mae = compute(grid_size, d, L, initial_sphere_radius)
        print(f"MAE(D={d})={mae}")
        mae_list.append(mae)

    # Check the law of large numbers over larger depths
    if len(args.depth_list) > 1:
        plt.plot(args.depth_list, mae_list)
        plt.xlabel("Depth")
        plt.ylabel("MAE")
        plt.title(f"Mean Absolute Error with various depth and L={L:03d}")
        plt.savefig(f"MAEL_{L:03d}.png")


def compute(grid_size, D, L, initial_sphere_radius):
    nn.clear_parameters()

    # 2D case
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    xx, yy = np.meshgrid(x, y)
    xy = np.asarray([xx.flatten(), yy.flatten()]).T

    x = nn.Variable.from_numpy_array(xy)
    y = implicit_network(x, D, feature_size=256, L=L,
                         initial_sphere_radius=initial_sphere_radius)
    y = y[:, 0]
    y.forward()

    # Plot
    plt.contourf(xx, yy, y.d.reshape(grid_size, grid_size))
    plt.colorbar()
    plt.contourf(xx, yy, y.d.reshape(grid_size, grid_size),
                 colors=('r',), levels=[-0.01, 0, 0.01])
    plt.title(f"Contour of SDF(x, y) = 0 (D={D:03d} and L={L:03d})")
    plt.savefig(f"sdf_contour_D{D:03d}_L{L:03d}.png")
    plt.clf()

    # error
    gt = np.sqrt(np.sum(xy ** 2, axis=1)) - 1
    mae = np.sum(np.abs(gt - y.d.flatten())) / len(gt)

    return mae


if __name__ == '__main__':
    import argparse

    nn.random.seed(414)

    description = "Implicit Network. This script is intended for checking" \
        "1) the geometric initialization by plotting contour." \
        "2) the law of large numbers."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--device-id', '-d', type=int, default="0")
    parser.add_argument('--depth-list', '-Ds', nargs="+",
                        type=int, default=[512])
    parser.add_argument('--layers', '-L', type=int, default=9)
    parser.add_argument('--initial-sphere-radius',
                        '-R', type=float, default=0.75)

    args = parser.parse_args()
    main(args)
