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

import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Any

import nnabla as nn
import nnabla.functions as F

import cv2

inch_to_mm = 25.4


class FitResolutionGate(Enum):
    FILL = 0
    OVER_SCREEN = 1


@dataclass
class Camera:
    """
    Camera is assumed to be oriended to negative z-axis as default.
    """

    # Intrinsic Parameter
    focal_length: float = 20              # in mm
    film_aperture_width: float = 0.980    # 35mm Full Aperture in inches
    film_aperture_height: float = 0.735   # 35mm Full Aperture in inches
    z_near: float = 1
    z_far: float = 1000
    image_width: int = 640
    image_height: int = 480
    k_fit_resolution_gate: FitResolutionGate = FitResolutionGate.OVER_SCREEN
    # Extrinsic Parameter
    camera_to_world: Any = None
    # Other
    z_orientation: int = -1  # {+1 or -1}

    def field_of_view(self, film_aperture):
        return 2 * 180 / np.pi * np.arctan((film_aperture * inch_to_mm / 2) / self.focal_length)

    def compute_screen(self):
        # fovs, top, right
        fov_h = self.field_of_view(self.film_aperture_width)
        fov_v = self.field_of_view(self.film_aperture_height)
        top = ((self.film_aperture_height * inch_to_mm / 2) /
               self.focal_length) * self.z_near
        right = ((self.film_aperture_width * inch_to_mm / 2) /
                 self.focal_length) * self.z_near

        # Scale if necessary
        x_scale = 1.0
        y_scale = 1.0
        film_aspect_ratio = self.compute_film_aspect_ratio()
        image_aspect_ratio = self.compute_image_aspect_ratio()
        if self.k_fit_resolution_gate == FitResolutionGate.FILL:
            if film_aspect_ratio > image_aspect_ratio:
                x_scale *= image_aspect_ratio / film_aspect_ratio
            else:
                y_scale *= film_aspect_ratio / image_aspect_ratio
        elif self.k_fit_resolution_gate == FitResolutionGate.OVER_SCREEN:
            if film_aspect_ratio > image_aspect_ratio:
                y_scale *= film_aspect_ratio / image_aspect_ratio
            else:
                x_scale *= image_aspect_ratio / film_aspect_ratio
        else:
            raise ValueError(
                f"k_fit_resolution_gate(={k_fit_resolution_gate}) must be in {list(FitResolutionGate)}.")
        top *= y_scale
        right *= x_scale
        bottom = -top
        left = -right

        # Screen (Image Plane)
        screen = Screen(left=left, right=right, top=top, bottom=bottom)
        return screen

    def compute_film_aspect_ratio(self):
        return 1.0 * self.film_aperture_width / self.film_aperture_height

    def compute_image_aspect_ratio(self):
        return 1.0 * self.image_width / self.image_height

    def compute_projection_matrix(self, screen):
        """Projection matrix (OpneGC style).
        """
        l = screen.left
        r = screen.right
        t = screen.top
        b = screen.bottom
        n = self.z_near
        f = self.z_far

        row0 = [2 * n / (r - l), 0, (r + l) / (r - l), 0]
        row1 = [0, 2 * n / (t - b), (t + b) / (t - b), 0]
        row2 = [0, 0, - (f + n) / (f - n), - 2 * f * n / (f - n)]
        row3 = [0, 0, self.z_orientation, 0]
        P = np.array([row0,
                      row1,
                      row2,
                      row3])
        return P

    def compute_intrinsic_inv(self, fov=None):
        """Inverse camera intrintic matrix.

        Transofmration formula is as follows: 

        Px = (2 * ((x + 0.5) / imageWidth) - 1) * tan(fov / 2 * M_PI / 180) * imageAspectRatio; 
        Py = (1 - 2 * ((y + 0.5) / imageHeight) * tan(fov / 2 * M_PI / 180)

        x: pixel x
        y: pixel y
        """
        W = self.image_width
        H = self.image_height
        fov = self.field_of_view(
            self.film_aperture_height) if fov is None else fov * np.pi / 180
        tan_fov_2 = np.tan(fov / 2)
        A = self.compute_image_aspect_ratio()

        row0 = [2 / W * tan_fov_2 * A, 0, (1 - W) / W * tan_fov_2 * A]
        row1 = [0, -2 / H * tan_fov_2, (H - 1) / H * tan_fov_2]
        row2 = [0, 0, self.z_orientation]
        K = np.array([row0,
                      row1,
                      row2])
        return K

    def compute_intrinsic(self, fov=None):
        return np.linalg.inv(self.compute_intrinsic_inv(fov))


@dataclass
class Screen:
    left: float = 0
    right: float = 0
    top: float = 0
    bottom: float = 0


def normalize(vertex):
    assert len(vertex) == 3
    vertex = vertex / np.sqrt(np.sum(vertex ** 2))
    return np.array([vertex[0], vertex[1], vertex[2]])


def look_at(F, T, tmp=np.array([0.0, 1.0, 0.0]), z_orientation=-1):
    """Return a camera-to-world transformation matrix.

    Camera is assumed to be oriended to negative z-axis.
    """
    if len(F) == 4 or len(T) == 4:
        F = F[:3]
        T = T[:3]
    forward = normalize(F - T) if z_orientation == -1 else normalize(T - F)
    right = np.cross(normalize(tmp), forward)
    up = np.cross(forward, right)
    M = np.stack([right, up, forward, F])
    H = np.array([0.0, 0.0, 0.0, 1.0])[:, np.newaxis]
    M = np.concatenate([M, H], axis=1)
    return M.T


class Light:
    """Base Light
    """

    def __init__(self,
                 color=nn.Variable.from_numpy_array([1., 1., 1.])):
        """
        Args:
          color (np.ndarray, nn.NdArray, nn.Variable): Light color.
        """
        if isinstance(color, np.ndarray):
            self.color = nn.Variable.from_numpy_array(color)
        elif isinstance(color, nn.NdArray):
            self.color = nn.Variable(color.shape)
            self.color.data = color
        elif isinstance(color, nn.Variable):
            self.color = color


class DistantLight(Light):
    """Distant Light
    """

    def __init__(self,
                 color=nn.Variable.from_numpy_array([1., 1., 1.]),
                 direction=nn.Variable.from_numpy_array([1., 1., 1.]),
                 ):
        """
        Args:
          color (np.ndarray, nn.NdArray, nn.Variable): Light color.
          direction (np.ndarray, nn.NdArray, nn.Variable): Direction of the light.
        """
        super(DistantLight, self).__init__(color)

        import nnabla.functions as F
        if isinstance(direction, np.ndarray):
            direction = nn.Variable.from_numpy_array(direction)
            self.direction = F.norm_normalization(direction, eps=1e-6)
        elif isinstance(direction, nn.NdArray):
            direction = nn.Variable(direction.shape)
            direction.data = direction
            self.direction = F.norm_normalization(direction, eps=1e-6)
        elif isinstance(direction, nn.Variable):
            self.direction = F.norm_normalization(direction, eps=1e-6)


class PointLight(Light):
    """Point Light
    """

    def __init__(self,
                 color=nn.Variable.from_numpy_array([1., 1., 1.]),
                 point=nn.Variable.from_numpy_array([0., 0., 0.]),
                 ):
        """
        Args:
          color (np.ndarray, nn.NdArray, nn.Variable): Light color.
          point (np.ndarray, nn.NdArray, nn.Variable): Position of the light.
        """
        super(PointLight, self).__init__(color)

        if isinstance(point, np.ndarray):
            self.point = nn.Variable.from_numpy_array(point)
        elif isinstance(point, nn.NdArray):
            self.point = nn.Variable(point.shape)
            self.point.data = point
        elif isinstance(point, nn.Variable):
            self.point = point


def lambert(normal, light_dir):
    """Lambertian (diffuse) lighting.

    Args:
      normal (nn.Variable): Normal of (B, R, 3) shape.
      light_dir (nn.Variable): Light direction of (3, ) shape.

    """
    base_axis = len(normal.shape) - 1
    cos = F.relu(F.affine(normal, light_dir.reshape(
        [3, 1]), base_axis=base_axis))
    return cos


def facing_ratio(normal, raydir):
    """Facing ratio lighting.

    Args:
      normal (nn.Variable): Normal of (B, R, 3) shape.
      raydir (nn.Variable): Ray direction of (B, R, 3) shape.

    """
    factor = F.relu(F.sum(normal * (-raydir), axis=1, keepdims=True))
    return factor


def load_K_Rt_from_P(P):
    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]  # intrinsic matrix
    R = out[1]  # world-to-camera matrix
    c = out[2]  # camera location

    K = K / K[2, 2]
    intrinsic = np.eye(4)
    intrinsic[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)  # [camera-to-world | camera location]
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (c[:3] / c[3])[:, 0]

    return intrinsic, pose


def generate_raydir_camloc(pose, intrinsic, xy):
    """
    pose: [B, 4, 4]
    intrinsic: [B, 3, 3]
    xy: [B, R, 2]

    x_p = K [R | t] x_w
    => x_p = K (R x_w + t)
    => x_w = R_inv K_inv x_p - R_inv t
    => x_w = normalize(x_w)
    """
    B, R, _ = xy.shape

    # Align dimensions
    R_c2w = pose[:, np.newaxis, :3, :3]
    camloc = pose[:, np.newaxis, :3, 3:4]
    K_inv = np.linalg.inv(intrinsic[:, np.newaxis, :, :])

    # Transform pixel --> camera --> world
    z = np.ones([B, R, 1])
    xyz_pixel = np.concatenate([xy, z], axis=-1)[:, :, :, np.newaxis]
    xyz_camera = np.matmul(K_inv, xyz_pixel)

    # Note: we do not need to subtract camloc since R_c2w does not contain it.
    xyz_world = np.matmul(R_c2w, xyz_camera)

    # Normalize
    xyz_world = xyz_world.reshape((B, R, 3))
    raydir = xyz_world / \
        np.sqrt(np.sum(xyz_world ** 2, axis=-1, keepdims=True))

    return raydir, camloc.reshape((B, 3))


def generate_all_pixels(W, H):
    x = np.arange(0, W)
    y = np.arange(0, H)
    xx, yy = np.meshgrid(x, y)
    xy = np.asarray([xx.flatten(), yy.flatten()]).T
    return xy


def create_monitor_path(data_path, monitor_path=""):
    base_path = data_path.rstrip("/").split("/")[-1]
    monitor_path = f"{monitor_path}_{base_path}"
    return monitor_path


if __name__ == '__main__':
    rng = np.random.RandomState(412)
    pose = rng.randn(1, 3, 4).astype(np.float32)
    intrinsic = np.asarray([[2.0, 0, 0.5], [0, 3.0, 0.5], [0, 0, 1]])[
                           np.newaxis].astype(np.float32)
    x = np.arange(0, 3)
    y = np.arange(0, 2)
    xx, yy = np.meshgrid(x, y)
    xy = (np.asarray([xx.flatten(), yy.flatten()]).T)[np.newaxis, ...]

    ray_dirs, cam_loc = generate_raydir_camloc(pose, intrinsic, xy)

    print(ray_dirs)
    print(cam_loc)
