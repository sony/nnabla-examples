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
from nnabla.ext_utils import get_extension_context
from nnabla.function import PythonFunction

import cv2
import time
from functools import partial

from helper import Camera, look_at, DistantLight, lambert


def bisection(x0, x1, implicit_function, max_post_itr):

    for i in range(max_post_itr):
        xm = (x0 + x1) * 0.5
        fm = implicit_function(xm)
        mp = F.greater_equal_scalar(fm, 0)
        mn = 1 - mp
        x0 = mp * xm + mn * x0  # f(x0) > 0
        x1 = mn * xm + mp * x1  # f(x1) < 0
    return x0, x1


def secant(x0, x1, implicit_function, max_post_itr, eps=1e-16):
    f0 = implicit_function(x0)  # > 0
    f1 = implicit_function(x1)  # < 0

    for i in range(max_post_itr):
        nu = f0 * (x1 - x0)
        de = f1 - f0
        mask0 = F.greater_scalar(F.abs(de), eps)
        mask1 = 1 - mask0
        nu = mask0 * nu + mask1 * 0
        de = mask0 * de + mask1 * 1

        xm = x0 - nu / de
        fm = implicit_function(xm)

        mp = F.greater_equal_scalar(fm, 0)
        mn = 1 - mp
        x0 = mp * xm + mn * x0
        f0 = mp * fm + mn * f0
        x1 = mn * xm + mp * x1
        f1 = mn * fm + mp * f1
    return x0, x1


class RayTrace(PythonFunction):

    def __init__(self, ctx, sdf_network, test=False, t_near=0, t_far=10,
                 sphere_trace_itr=50, ray_march_points=100, n_chunks=10,
                 max_post_itr=10, post_method="secant", eps=5e-5):

        super(RayTrace, self).__init__(ctx)

        self.sdf = sdf_network
        self.test = test
        self.t_init = t_near
        self.t_near = t_near
        self.t_far = t_far
        self.sphere_trace_itr = sphere_trace_itr
        self.ray_march_points = ray_march_points
        self.n_chunks = n_chunks
        self.max_post_itr = max_post_itr
        self.eps = eps
        if post_method == "secant":
            self.post_method = partial(
                secant, implicit_function=sdf_network, max_post_itr=max_post_itr)
        elif post_method == "bisection":
            self.post_method = partial(
                bisection, implicit_function=sdf_network, max_post_itr=max_post_itr)
        else:
            self.post_method = lambda x0, x1: (x0, x1)

    @property
    def name(self):
        return "RayTrace"

    def min_outputs(self):
        return 5

    def setup_impl(self, inputs, outputs):
        camloc = inputs[0]
        raydir = inputs[1]
        x_hit = outputs[0]
        mask_hit = outputs[1]
        dists = outputs[2]
        mask_pin = outputs[3]
        mask_pout = outputs[4]

        B, R, _ = raydir.shape

        x_hit.reset_shape((B, R, 3), True)
        mask_hit.reset_shape((B, R, 1), True)
        dists.reset_shape((B, R, 1), True)
        mask_pin.reset_shape((B, R, 1), True)
        mask_pout.reset_shape((B, R, 1), True)

    def forward_impl(self, inputs, outputs):
        # This auto forward is needed for weight normalization auto forward
        with nn.auto_forward():
            self._forward_impl(inputs, outputs)

    def _forward_impl(self, inputs, outputs):
        # inputs[0]: camera location (B, 3)
        # inputs[1]: ray direction (B, R, 3)
        # inputs[2]: mask_obj (B, R, 1)
        # outputs[0]: intersection points (B, R, 3)
        # outputs[1]: mask of intersection
        # outputs[2]: distance after unit sphere intersection (B, R, 1)
        #   - 1. between x_hit and camloc for P_in (= hit and mask_obj)
        #   - 2. as t_argmin for P_out (= ~P_in)
        # outputs[3]: mask of P_in (B, R, 1)
        # outputs[3]: mask of P_out (B, R, 1)

        # All variables becomes [B, R, 3] --> [B * R, {1, 3}] at the beginning
        # sdf_network: [B * R, 3] --> [B * R, 1]

        # Settings
        test = self.test
        t_near = self.t_near
        t_far = self.t_far
        sdf = self.sdf
        sphere_trace_itr = self.sphere_trace_itr
        ray_march_points = self.ray_march_points
        n_chunks = self.n_chunks
        max_post_itr = self.max_post_itr
        eps = self.eps

        # Ndarray
        camloc = inputs[0].data
        raydir = inputs[1].data
        mask_obj = inputs[2].data
        B, R, _ = raydir.shape
        N = ray_march_points
        camloc = F.broadcast(camloc[:, np.newaxis, :], (B, R, 3))
        camloc = F.reshape(camloc, (B * R, 3))
        raydir = F.reshape(raydir, (B * R, 3))
        mask_obj = F.reshape(mask_obj, (B * R, 1)) if not test else mask_obj

        # Unit sphare intersection
        t_start, t_finish, mask_us = \
            self.unit_sphere_intersection(camloc, raydir)

        # Bidirectional sphere tracing
        x_hit_st0, t_f, t_b, mask_st = \
            self.bidirectional_sphere_trace(camloc, raydir,
                                            t_start, t_finish)

        mask_obj = 1 if test else mask_obj

        # Ray marching
        x_hit_rm0, x_hit_rm1, mask_rm = self.ray_march(
            camloc, raydir, t_f, t_b, N, n_chunks)
        x_hit_rm, _ = self.post_method(x_hit_rm0, x_hit_rm1)

        x_hit = mask_st * x_hit_st0 + (1 - mask_st) * mask_rm * x_hit_rm
        mask_hit = mask_st + (1 - mask_st) * mask_rm

        if test:
            x_hit = F.reshape(x_hit, (B, R, 3))
            mask_hit = F.reshape(mask_hit, (B, R, 1))
            outputs[0].data.copy_from(x_hit)
            outputs[1].data.copy_from(mask_hit)
            return

        # Mask pin/pout
        mask_pin = mask_us * mask_hit * mask_obj
        mask_pout = mask_us * (1 - mask_hit * mask_obj)

        # Dists
        t_argmin = self.ray_march(
            camloc, raydir, t_start, t_finish, N, n_chunks, True)
        dists = F.norm(camloc - x_hit, axis=x_hit.ndim - 1, keepdims=True)
        dists = mask_pin * dists + mask_pout * t_argmin

        # Outputs
        x_hit = F.reshape(x_hit, (B, R, 3))
        mask_hit = F.reshape(mask_hit, (B, R, 1))
        dists = F.reshape(dists, (B, R, 1))
        mask_pin = F.reshape(mask_pin, (B, R, 1))
        mask_pout = F.reshape(mask_pout, (B, R, 1))
        outputs[0].data.copy_from(x_hit)
        outputs[1].data.copy_from(mask_hit)
        outputs[2].data.copy_from(dists)
        outputs[3].data.copy_from(mask_pin)
        outputs[4].data.copy_from(mask_pout)

    def unit_sphere_intersection(self, camloc, raydir):
        BR, _ = raydir.shape
        a = 1.0  # raydir is already normalized
        b = 2.0 * F.batch_matmul(F.reshape(camloc, (BR, 1, 3)),
                                 F.reshape(raydir, (BR, 3, 1)))
        c = F.batch_matmul(F.reshape(camloc, (BR, 1, 3)),
                           F.reshape(camloc, (BR, 3, 1))) - 1.0
        D = b ** 2 - 4 * a * c
        mask = F.reshape(F.greater_scalar(D, 0.0), (BR, 1))

        b = F.reshape(b, (BR, 1))
        D = F.reshape(D, (BR, 1))

        D = mask * D
        D_sqrt = D ** 0.5
        t_start = - (b + D_sqrt) / (2 * a)
        t_finish = - (b - D_sqrt) / (2 * a)

        t_start = t_start * mask + self.t_near * (1 - mask)
        t_finish = t_finish * mask + self.t_far * (1 - mask)

        return t_start, t_finish, mask

    def bidirectional_sphere_trace(self, camloc, raydir, t_start, t_finish):
        t_f = F.identity(t_start)
        x_f = camloc + t_f * raydir
        s_f = self.sdf(x_f)
        mask_hit_eps_f = 0 * F.identity(t_f)

        t_b = F.identity(t_finish)
        x_b = camloc + t_b * raydir
        s_b = self.sdf(x_b)
        mask_hit_eps_b = 0 * F.identity(t_b)

        for i in range(self.sphere_trace_itr - 1):
            # Forward direction
            mask_hit_eps_f_i = F.less_equal_scalar(F.abs(s_f), self.eps)
            mask_hit_eps_f += (1 - mask_hit_eps_f) * mask_hit_eps_f_i
            t_f += (1 - mask_hit_eps_f) * s_f
            x_f = camloc + t_f * raydir

            s_f_prev = F.identity(s_f)
            s_f = self.sdf(x_f)
            mask_pos_f_prev = (1 - mask_hit_eps_f) * \
                F.greater_scalar(s_f_prev, 0)
            mask_neg_f = (1 - mask_hit_eps_f) * F.less_scalar(s_f, 0)
            mask_revert_f = mask_pos_f_prev * mask_neg_f
            t_f -= mask_revert_f * s_f_prev
            s_f = mask_revert_f * s_f_prev + (1 - mask_revert_f) * s_f

            # Backward direction
            mask_hit_eps_b_i = F.less_equal_scalar(F.abs(s_b), self.eps)
            mask_hit_eps_b += (1 - mask_hit_eps_b) * mask_hit_eps_b_i
            t_b -= (1 - mask_hit_eps_b) * s_b
            x_b = camloc + t_b * raydir

            s_b_prev = F.identity(s_b)
            s_b = self.sdf(x_b)
            mask_pos_b_prev = (1 - mask_hit_eps_b) * \
                F.greater_scalar(s_b_prev, 0)
            mask_neg_b = (1 - mask_hit_eps_b) * F.less_scalar(s_b, 0)
            mask_revert_b = mask_pos_b_prev * mask_neg_b
            t_b += mask_revert_b * s_b_prev
            s_b = mask_revert_b * s_b_prev + (1 - mask_revert_b) * s_b

            ## print("s_f neg", np.sum(s_f.data < 0))
            ## print("s_b neg", np.sum(s_b.data < 0))

        # Fine grained start/finish points
        t_f0 = t_f
        t_f1 = t_f + mask_revert_f * s_f_prev
        x_hit_st0 = camloc + t_f0 * raydir
        ## x0, x1 = self.post_method(x_hit_st0, camloc + t_f1 * raydir)
        ## t_f0 = F.norm((x0 - camloc), axis=(x0.ndim - 1), keepdims=True)
        ## t_f1 = F.norm((x1 - camloc), axis=(x1.ndim - 1), keepdims=True)
        mask_hit_f1b = mask_revert_f * F.less(t_f1, t_b)
        t_b = t_f1 * mask_hit_f1b + t_b * (1 - mask_hit_f1b)

        # Reverse the opposite case
        mask_fb = F.less(t_f, t_b)
        t_f = t_f * mask_fb + t_start * (1 - mask_fb)
        t_b = t_b * mask_fb + t_finish * (1 - mask_fb)

        return x_hit_st0, t_f, t_b, mask_hit_eps_f

    def ray_march(self, camloc, raydir, t0, t1, N, n_chunks, t_argmin=False):
        # Points computation
        BR, _ = t0.shape
        t0 = F.reshape(t0, (BR, 1, 1))
        t1 = F.reshape(t1, (BR, 1, 1))
        camloc = F.reshape(camloc, (BR, 1, 3))
        raydir = F.reshape(raydir, (BR, 1, 3))
        step = (t1 - t0) / (N-1)
        intervals = F.reshape(F.arange(0, N), (1, N, 1))
        ts = t0 + step * intervals
        points = camloc + ts * raydir
        points = F.reshape(points, (BR * N, 3))

        # SDF computation
        sdf_points = []
        batch = (BR * N) // n_chunks
        for r in range(0, BR * N, batch):
            sdf_points.append(self.sdf(points[r:r+batch, :]))
        sdf_points = F.reshape(F.concatenate(*sdf_points, axis=0), (BR, N, 1)) if n_chunks != 1 else \
            F.reshape(sdf_points[0], (BR, N, 1))

        # t_argmin computation
        if t_argmin:
            idx_min = F.min(sdf_points, axis=1, keepdims=True, only_index=True)
            t_argmin = F.reshape(
                F.gather(ts, idx_min, axis=1, batch_dims=1), (BR, 1))
            return t_argmin

        # Intersection check
        points = F.reshape(points, (BR, N, 3))
        sdf_pos = F.greater_equal_scalar(sdf_points[:, :-1, :], 0)
        sdf_neg = F.less_equal_scalar(sdf_points[:, 1:, :], 0)
        mask_hit = sdf_pos * sdf_neg

        decreasing_consts = F.reshape(F.arange(N, 1, -1), (1, N - 1, 1))
        vals = mask_hit * decreasing_consts
        idx_max = F.max(vals, axis=1, only_index=True)

        points = points[:, :-1, :]
        x_hit = F.gather(points, idx_max, axis=1, batch_dims=1)
        x_hit = F.reshape(x_hit, (BR, 3))
        mask_hit = F.greater_scalar(F.sum(mask_hit, axis=1), 0)
        mask_hit = F.reshape(mask_hit, (BR, 1))

        x_hit_rm0 = x_hit
        step = F.reshape(step, (BR, 1))
        raydir = F.reshape(raydir, (BR, 3))
        x_hit_rm1 = x_hit_rm0 + step * raydir

        return x_hit_rm0, x_hit_rm1, mask_hit

    def backward_impl(self, inputs, outputs, propagate_down, accum):
        if not propagate_down[0] and not propagate_down[1]:
            return

        # Accum is zero addition, thus do not care.
        if propagate_down[0]:
            if not accum[0]:
                inputs[0].grad.fill(0)

        if propagate_down[1]:
            if not accum[1]:
                inputs[1].grad.fill(1)


def ray_trace(sdf_network, camloc, raydir, mask_obj=None, test=False, t_near=0, t_far=4,
              sphere_trace_itr=50, ray_march_points=100, n_chunks=10,
              max_post_itr=10, post_method="secant", eps=5e-5,
              ctx=None):
    func = RayTrace(ctx, sdf_network, test, t_near=t_near, t_far=t_far,
                    sphere_trace_itr=sphere_trace_itr, ray_march_points=ray_march_points,
                    n_chunks=n_chunks,
                    max_post_itr=max_post_itr,
                    post_method=post_method,
                    eps=eps)
    mask_obj = mask_obj if mask_obj is not None else nn.Variable()  # dummy
    return func(camloc, raydir, mask_obj)


def main(args):
    from network import implicit_network

    # Setting
    # nn.set_auto_forward(True)
    ctx = get_extension_context('cudnn', device_id=args.device_id)
    nn.set_default_context(ctx)
    D = args.depth
    L = args.layers
    W = args.width
    H = args.height
    R = H * W
    z_orientation = 1

    # Camera parameters
    camera = Camera(image_width=W, image_height=H, z_orientation=z_orientation)
    camloc = np.array([0.75, 0.5, 1])
    camloc = (camloc / np.sum(camloc ** 2) ** 0.5) * 2
    to = np.array([0, 0, 0])
    Rt_inv = look_at(camloc, to, z_orientation=z_orientation)
    R_inv = Rt_inv[:3, :3]
    fov = 90
    K_inv = camera.compute_intrinsic_inv(fov)

    # Rays
    x, y = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    xy = np.asarray([x.flatten(), y.flatten()])
    xy1 = np.concatenate([xy, np.ones(R)[np.newaxis, :]])
    raydir = R_inv.dot(K_inv.dot(xy1))
    raydir = raydir / np.sum(raydir ** 2, axis=0) ** 0.5
    raydir = raydir.transpose((1, 0))

    # Network
    camloc = nn.Variable.from_numpy_array(camloc[np.newaxis, ...])
    raydir = nn.Variable.from_numpy_array(raydir[np.newaxis, ...])
    sdf_net = partial(implicit_network, D=D, L=L,
                      initial_sphere_radius=args.initial_sphere_radius)
    sdf_net0 = sdf_net

    def sdf_net0(x):
        out = sdf_net(x)
        sdf = out[..., 0][..., np.newaxis]
        return sdf

    # Sphere trace
    t_near = args.t_near
    t_far = args.t_far
    sphere_trace_itr = args.sphere_trace_itr
    ray_march_points = args.ray_march_points
    n_chunks = args.n_chunks
    max_post_itr = args.max_post_itr
    post_method = args.post_method
    eps = args.eps
    st = time.time()
    x_hit, mask_hit, dists, _, _ = ray_trace(sdf_net0, camloc, raydir, test=True,
                                             t_near=t_near, t_far=t_far,
                                             sphere_trace_itr=sphere_trace_itr,
                                             ray_march_points=ray_march_points,
                                             n_chunks=n_chunks,
                                             max_post_itr=max_post_itr,
                                             post_method=post_method, eps=eps)

    x_hit.need_grad = False
    dists.need_grad = False
    mask_hit.need_grad = False

    x_curr = x_hit
    F.sink(*[x_curr, mask_hit]).forward(clear_buffer=False)
    # Lighting
    x_curr = x_curr.get_unlinked_variable(need_grad=True)
    sdf = sdf_net0(x_curr)
    normal = nn.grad([sdf], [x_curr])[0]
    normal = F.norm_normalization(normal, axes=normal.ndim - 1, eps=1e-24)
    dlight = DistantLight()
    cos = lambert(normal, dlight.direction.reshape([3, 1])).reshape((1, H, W))
    mask_hit = mask_hit.get_unlinked_variable(need_grad=False)
    mask_hit = F.reshape(mask_hit, (1, H, W))
    mask_hit = F.broadcast(mask_hit, (3, H, W))
    image = mask_hit * 255.0 * cos
    image.forward(clear_buffer=True)

    cv2.imwrite(f"sphere_{W}x{H}_sti{sphere_trace_itr:03d}_mpi{max_post_itr:03d}_{args.post_method}.png",
                image.d.transpose(1, 2, 0))
    print(
        f"Bidirectional sphere trace/ray march (W={W}, H={H}): {time.time() - st} [s]")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Sphere Tracinig.")
    parser.add_argument('--device-id', '-d', type=int, default="0")
    parser.add_argument('--depth', '-D', type=int, default=200)
    parser.add_argument('--layers', '-L', type=int, default=9)
    parser.add_argument('--width', '-W', type=int, default=240)
    parser.add_argument('--height', '-H', type=int, default=180)
    parser.add_argument('--t-near', '-tf', type=int, default=0)
    parser.add_argument('--t-far', '-tn', type=int, default=4)
    parser.add_argument('--sphere-trace-itr', '-sti', type=int, default=10)
    parser.add_argument('--ray-march-points', '-N', type=int, default=100)
    parser.add_argument('--n-chunks', '-NC', type=int, default=10)
    parser.add_argument('--max-post-itr', '-mpi', type=int, default=8)
    parser.add_argument('--eps', type=float, default=5e-5)
    parser.add_argument('--initial-sphere-radius', type=float, default=0.75)
    parser.add_argument('--post-method', type=str, default="bisection")
    parser.add_argument('--fov', type=float, default=90)
    args = parser.parse_args()
    main(args)
