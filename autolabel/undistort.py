import numpy as np
import sys
import cv2


class ImageUndistorter:
    """Undistorts images

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

        self._compute_source_to_target_mapping()

    def _compute_source_to_target_mapping(self) -> None:
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self._K, self._D, np.eye(3), self._K, (self._W, self._H),
            cv2.CV_32FC2)

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        return cv2.remap(image, self.map1, self.map2, cv2.INTER_NEAREST)
