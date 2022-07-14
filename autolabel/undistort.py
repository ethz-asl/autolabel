import numpy as np
import sys
import torch
import torch_scatter
from numba import jit
from typing import Tuple

EPSILON = sys.float_info.epsilon


@jit(nopython=True)
def _apply_camera_distortion(D: np.ndarray, u: np.ndarray,
                             v: np.ndarray) -> np.ndarray:
    """Converted from instant-ngp/include/neural-graphics-primitives/
    common_device.cuh.
    """
    k_1, k_2, p_1, p_2 = D

    u2 = u * u
    uv = u * v
    v2 = v * v
    r2 = u2 + v2
    radial = k_1 * r2 + k_2 * r2 * r2

    dx = np.array([
        u * radial + 2 * p_1 * uv + p_2 * (r2 + 2 * u2),
        v * radial + 2 * p_2 * uv + p_1 * (r2 + 2 * v2)
    ])

    return dx


@jit(nopython=True)
def _iterative_camera_undistortion(
        K: np.ndarray, D: np.ndarray, u: np.ndarray,
        v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Converted from instant-ngp/include/neural-graphics-primitives/
    common_device.cuh.

    Parameters for Newton iteration using numerical differentiation with
    central differences, 100 iterations should be enough even for complex
    camera models with higher order terms.
    """
    kNumIterations = 100
    kMaxStepNorm = 1.e-10
    kRelStepSize = 1.e-6

    f_x = K[0, 0]
    f_y = K[1, 1]
    c_x = K[0, 2]
    c_y = K[1, 2]

    ray_u = (u - c_x) / f_x
    ray_v = (v - c_y) / f_y

    J = np.zeros((np.int(2), np.int(2)), dtype=np.float64)

    x0 = np.array([ray_u, ray_v])
    x = np.array([ray_u, ray_v])

    for _ in range(kNumIterations):
        step0 = max(EPSILON, np.abs(kRelStepSize * x[0]))
        step1 = max(EPSILON, np.abs(kRelStepSize * x[1]))
        dx = _apply_camera_distortion(D, x[0], x[1])
        dx_0b = _apply_camera_distortion(D, x[0] - step0, x[1])
        dx_0f = _apply_camera_distortion(D, x[0] + step0, x[1])
        dx_1b = _apply_camera_distortion(D, x[0], x[1] - step1)
        dx_1f = _apply_camera_distortion(D, x[0], x[1] + step1)
        J[0, 0] = 1 + (dx_0f[0] - dx_0b[0]) / (2 * step0)
        J[0, 1] = (dx_1f[0] - dx_1b[0]) / (2 * step1)
        J[1, 0] = (dx_0f[1] - dx_0b[1]) / (2 * step0)
        J[1, 1] = 1 + (dx_1f[1] - dx_1b[1]) / (2 * step1)
        step_x = np.linalg.inv(J) @ (x + dx - x0)
        x -= step_x
        if (np.linalg.norm(step_x)**2 < kMaxStepNorm):
            break

    ray_u_transformed = x[0]
    ray_v_transformed = x[1]

    u_transformed = ray_u_transformed * f_x + c_x
    v_transformed = ray_v_transformed * f_y + c_y

    return u_transformed, v_transformed


class ImageUndistorter:
    """Implements functions to undistort images given OpenCV intrinsics
    estimated by COLMAP. This way, the undistorted images can be used with the
    estimated poses by NGP assuming an ideal pinhole camera model, i.e., without
    distortion.

    Args:
        K (np.ndarray): Intrinsics matrix. Shape: (3, 3).
        D (np.ndarray): Distortion coefficients (k_1, k_2, p_1, p_2) from the
            "OPENCV" COLMAP model. Shape: (4, ).
        H (int): Image height.
        W (int): Image width.
    """

    def __init__(self, K: np.ndarray, D: np.ndarray, H: int, W: int):
        self._K = K
        self._D = D
        self._H = H
        self._W = W

        # Compute the undistortion map for each source pixel.
        self._compute_source_to_target_mapping()
        # For each target pixel, select the source pixel based on the
        # undistortion map and nearest neighbors (i.e., perform nearest-neighbor
        # interpolation).
        self._compute_best_uv()

    def _compute_source_to_target_mapping(self) -> None:
        """Computes the mapping from a pixel coordinate (source) to its
        corresponding coordinate after undistortion (target), with rounding to
        the nearest integer coordinate. It also computes the (square) distance
        between the rounded coordinate and the actual (floating-point)
        coordinate returned by the undistortion. This distance is required to
        later select the source pixel for each target pixel when doing
        nearest-neighbor interpolation.

        Args:
            None.

        Returns:
            None.
        """
        u_transformed = np.empty([self._H, self._W])
        v_transformed = np.empty([self._H, self._W])

        res = np.array([
            _iterative_camera_undistortion(self._K, self._D, u, v)
            for (u, v) in np.stack(
                np.meshgrid(np.arange(self._W), np.arange(self._H))).reshape(
                    2, -1).transpose()
        ])

        uv = np.stack(np.meshgrid(np.arange(self._W),
                                  np.arange(self._H))).reshape(2,
                                                               -1).transpose()
        u_transformed[uv[..., 1], uv[..., 0]] = res[..., 0]
        v_transformed[uv[..., 1], uv[..., 0]] = res[..., 1]

        self._u_transformed_rounded = np.round(u_transformed).astype(int)
        self._v_transformed_rounded = np.round(v_transformed).astype(int)

        self._transformed_sq_dist_to_pix = (
            (self._u_transformed_rounded - u_transformed)**2 +
            (self._v_transformed_rounded - v_transformed)**2)

    def _compute_best_uv(self) -> None:
        """For each pixel in target image, select the pixel in the source image
        that results in the closest pixel after undistortion (i.e., perform
        nearest-neighbor interpolation).

        Args:
            None.

        Returns:
            None.
        """
        # Flatten.
        uv_transformed_rounded = (self._v_transformed_rounded * self._W +
                                  self._u_transformed_rounded).reshape(-1)
        self._transformed_sq_dist_to_pix = (
            self._transformed_sq_dist_to_pix.reshape(-1))

        # For each target coordinates that has at least a corresponding source
        # coordinate, find the source coordinate that when transformed is
        # closest to the target coordinate.
        uv_where = torch_scatter.scatter_min(
            src=torch.from_numpy(
                self._transformed_sq_dist_to_pix).to(device='cuda'),
            index=torch.from_numpy(uv_transformed_rounded).to(device='cuda'),
            dim_size=self._H * self._W)[1].cpu().numpy().reshape(
                self._H, self._W)

        best_u = -np.ones([self._H, self._W])
        best_v = -np.ones([self._H, self._W])

        best_u[uv_where != (
            self._H *
            self._W)] = uv_where[uv_where != (self._H * self._W)] % self._W
        best_v[uv_where != (
            self._H *
            self._W)] = uv_where[uv_where != (self._H * self._W)] // self._W

        self._best_uv_for_given_output_pixel = np.stack([best_u, best_v],
                                                        axis=-1)

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        r"""Undistorts an input image based on the distortion coefficients
        provided and using nearest-neighbor interpolation.

        Args:
            image (np.ndarray): Image to interpolate. Shape:
                (H, W, num_channels).

        Returns:
            Interpolated image, with the same shape as the input image.
        """
        output_image = np.zeros_like(image)

        has_output_pixel_corr_pixel = self._best_uv_for_given_output_pixel[
            ..., 0] != -1
        v_where, u_where = np.where(
            self._best_uv_for_given_output_pixel[..., 0] != -1)
        output_image[v_where, u_where] = image[
            self._best_uv_for_given_output_pixel[has_output_pixel_corr_pixel][
                ..., 1].astype(np.long),
            self._best_uv_for_given_output_pixel[has_output_pixel_corr_pixel][
                ..., 0].astype(np.long)]

        return output_image
