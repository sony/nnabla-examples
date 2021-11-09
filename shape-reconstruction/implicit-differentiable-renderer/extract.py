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
import numpy as np
from nnabla.ext_utils import get_extension_context

import os
from functools import partial
from tqdm import tqdm

import trimesh
from skimage import measure

from network import sdf_net


def create_mesh_from_volume(volume, spacing, gradient_direction="ascent"):
    spacing = [np.max(spacing)] * 3
    verts, faces, normals, values = measure.marching_cubes_lewiner(volume,
                                                                   0.0,
                                                                   spacing=spacing,
                                                                   gradient_direction=gradient_direction)
    mesh = trimesh.Trimesh(verts, faces, normals)
    return mesh


def compute_pts_vol(model, mins, maxs, step, sub_batch_size, bias=None, V=None):

    x = np.arange(mins[0], maxs[0], step).astype(np.float32)
    y = np.arange(mins[1], maxs[1], step).astype(np.float32)
    z = np.arange(mins[2], maxs[2], step).astype(np.float32)
    X, Y, Z = np.meshgrid(x, y, z)
    grid_shape = X.shape
    print(f"Grid shape for (x, y, z) = {grid_shape}")

    X = X.reshape(-1)
    Y = Y.reshape(-1)
    Z = Z.reshape(-1)
    pts = np.stack((X, Y, Z), axis=1)

    vol = []
    for b in tqdm(range(0, pts.shape[0], sub_batch_size),
                  desc="compute-volume"):
        p = pts[b:b+sub_batch_size, :]
        p = (p - bias) @ V.T if V is not None else p
        #p = p @ V.T if V is not None else p
        v = model(nn.NdArray.from_numpy_array(p))
        v = v.data.copy().reshape(-1)
        vol.append(v)
    pts = pts.reshape((-1, 3))

    s0, s1, s2 = grid_shape
    vol = np.concatenate(vol).reshape((s0, s1, s2)).transpose((0, 1, 2))
    return pts, vol


def create_largest_mesh(mesh):
    meshes = mesh.split(only_watertight=False)
    areas = np.array([m.area for m in meshes], dtype=np.float)
    mesh = meshes[areas.argmax()]
    return mesh


def main(args):
    conf = args.conf

    # Context
    ctx = get_extension_context("cudnn", device_id=args.device_id)
    nn.set_default_context(ctx)
    nn.set_auto_forward(True)

    # Network
    nn.load_parameters(args.model_load_path)

    # Extract rough mesh
    mins = np.asarray([-conf.bounding_box_size] * 3)
    maxs = np.asarray([conf.bounding_box_size] * 3)
    step = 2 * conf.bounding_box_size / 100
    pts, vol = compute_pts_vol(partial(sdf_net, conf=conf),
                               mins, maxs, step,
                               conf.sub_batch_size)
    mesh_low = create_mesh_from_volume(
        vol, [step] * 3, conf.gradient_direction)
    mesh_large = create_largest_mesh(mesh_low)

    # Samples
    points_all = trimesh.sample.sample_surface(mesh_low, 10000)[0]
    points_large = trimesh.sample.sample_surface(mesh_large, 10000)[0]

    # Centering and rotate
    mean = np.mean(points_large, axis=0, keepdims=True)
    cov = (points_large - mean).transpose((1, 0)) @ (points_large - mean)
    _, V = np.linalg.eig(cov)
    points_all = (points_all - mean) @ V
    points_large = (points_large - mean) @ V

    # Create scale and bias
    mins_all = np.asarray([min(m0, m1) for m0, m1 in zip(
        points_all.min(axis=0), points_large.min(axis=0))])
    maxs_all = np.asarray([max(m0, m1) for m0, m1 in zip(
        points_all.max(axis=0), points_large.max(axis=0))])
    idx_max_length = np.argmax(maxs_all - mins_all)
    m = mins_all[idx_max_length]
    M = maxs_all[idx_max_length]
    l0 = -conf.bounding_box_size
    l1 = conf.bounding_box_size
    scale = (l1 - l0) / (M - m)
    bias = l0 - scale * m

    # Extract fine mesh
    mins_large = points_large.min(axis=0)
    maxs_large = points_large.max(axis=0)
    min_length = np.min(maxs_large - mins_large)
    grid_step = min_length * scale / conf.grid_size
    print(f"grid_step = {grid_step}")
    mins = scale * mins_large + bias
    maxs = scale * maxs_large + bias
    pts, vol = compute_pts_vol(partial(sdf_net, conf=conf),
                               mins, maxs, grid_step,
                               conf.sub_batch_size,
                               bias, V)
    mesh = create_mesh_from_volume(
        vol, [grid_step] * 3, conf.gradient_direction)
    mesh = create_largest_mesh(mesh)

    # Save
    dirname, pname = args.model_load_path.split("/")
    fname = os.path.splitext(pname)[0]
    fpath = f"{dirname}/{fname}_mesh.ply"
    nn.logger.info(f"Saving the mesh file to {fpath}")
    mesh.export(fpath, "ply")


if __name__ == '__main__':
    import argparse
    from ruamel.yaml import YAML
    from collections import namedtuple

    parser = argparse.ArgumentParser(
        description="Implicit Differentiable Renderer Training.")
    parser.add_argument('--device-id', '-d', type=int, default="0")
    parser.add_argument('--model-load-path', type=str, required=True)
    parser.add_argument('--config', type=str,
                        default="conf/default.yaml", required=True)

    args = parser.parse_args()
    with open(args.config, "r") as f:
        conf = YAML(typ='safe').load(f)
        conf = namedtuple("Conf", conf)(**conf)
    args.conf = conf

    main(args)
