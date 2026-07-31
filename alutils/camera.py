# Numpy
import numpy as np
from numpy.typing import NDArray

# Utils
from .pose import Pose
from .base import homogenized, dehomogenized


def get_homogeneous_pixels_array(image_shape: tuple[int]) -> NDArray:
    """
    Create an array of homogeneous coordinates `(u, v, 1)` for every pixel.

    Args:
        image_shape: `tuple[int]` image dimensions `(H, W)`

    Returns:
        pixels: `NDArray(H, W, 3)` homogeneous pixel coordinates

    Raises:
        ValueError: `image_shape` does not contain exactly two elements
    """

    if not len(image_shape) == 2:
        raise ValueError(
            "The provided shape must have 2 elements, i.e. (H, W), not "
            f"{image_shape}."
        )

    # (H, W, 3) homogeneous 2D pixel coordinates
    pixels = np.ones((*image_shape, 3), dtype=int)
    pixels[:, :, 0] = np.arange(image_shape[1])[None, :]  # u values
    pixels[:, :, 1] = np.arange(image_shape[0])[:, None]  # v values

    return pixels


class PinholeCamera:

    def __init__(self, K: NDArray, pose: Pose = Pose()):
        """
        Default constructor of the `PinholeCamera` class.

        Args:
            K: `NDArray(3, 3)` camera intrinsics matrix

        Optional Args:
            pose: `Pose` camera extrinsics represented by the transformation
                  from the camera frame to the world frame. The translation
                  `t = (tx, ty, tz)` gives the camera coordinates in the world
                  frame. Default is the identity pose.
        """
        self.__pose = pose
        self.set_intrinsics(K)

    def set_intrinsics(self, K: NDArray):
        """
        Set the camera intrinsics matrix.

        Args:
            K: `NDArray(3, 3)` camera intrinsics matrix

        Requirements:
            K is a 3x3 invertible matrix
        """
        assert(K.shape == (3, 3))
        self.__K = K
        self.__K_inv = np.linalg.inv(K)
        self._compute_projection_matrix()

    def set_pose(self, pose):
        """
        Set the camera pose.

        Args:
            pose: `Pose` camera extrinsics represented by the transformation
                  from the camera frame to the world frame
        """
        self.__pose = pose
        self._compute_projection_matrix()

    def _compute_projection_matrix(self):
        """Compute and store the camera projection matrix."""
        self.__P = self.K @ self.__pose.inverse.Rt

    @property
    def K(self) -> NDArray:
        """
        `NDArray(3,3)`: The camera intrinsics matrix.
        """
        return self.__K

    @property
    def K_inv(self) -> NDArray:
        """
        `NDArray(3,3)`: The inverse camera intrinsics matrix.
        """
        return self.__K_inv

    @property
    def P(self) -> NDArray:
        """
        `NDArray(3,4)`: The camera projection matrix.
        """
        return self.__P

    @property
    def pose(self) -> Pose:
        """
        `Pose`: The pose of the camera.
        """
        return self.__pose


    def backproject_to_world_using_depth(
        self,
        coordinates: NDArray,
        depth: NDArray,
    ) -> NDArray:
        """
        Backproject 2D pixel coordinates to 3D world points using depth.

        That is, `3D world points = Rt @ depth @ K^-1 @ coordinates`.

        Args:
            coordinates: `NDArray(..., 2)` or `NDArray(..., 3)` pixel
                         coordinates
            depth: `NDArray(...)` depth for each pixel coordinate

        Returns:
            points_world: `NDArray(..., 3)` 3D points in the world frame

        Note:
            Pixels with negative depth are not supported and must be rejected
            before using this method.
        """

        # Make coordinates homogeneous
        if coordinates.shape[-1] == 2:
            coordinates = homogenized(coordinates)

        if not coordinates.shape[:-1] == depth.shape:
            raise ValueError(
                f"The shape of coordinates {coordinates.shape} and depth "
                f"{depth.shape} do not match. They should be in the forms "
                "(d_0, ..., d_N, 3) and (d_0, ..., d_N)."
            )

        # Get normalized homogeneous pixel coordinates
        normalized_pixels = (
            self.K_inv @ coordinates[..., None]
        )[..., 0]  # (..., 3)

        # Project using depth
        points_camera = depth[..., None] * normalized_pixels  # (..., 3)

        # Apply extrinsics
        points_world = (
            self.pose.Rt @ homogenized(points_camera)[..., None]
        )[..., 0]  # (..., 3)

        return points_world

    def backproject_camera_frame(self, coordinates: NDArray) -> NDArray:
        """
        Backproject pixel coordinates to normalized camera-frame coordinates.

        Args:
            coordinates: `NDArray(..., 2)` or `NDArray(..., 3)` image or
                         homogeneous pixel coordinates. For 2D coordinates,
                         the homogeneous scale is assumed to be 1.

        Returns:
            points_camera: `NDArray(..., 2)` normalized camera-frame
                           coordinates
        """

        # Make coordinate homogeneous
        if coordinates.shape[-1] == 2:
            coordinates = homogenized(coordinates)
        assert coordinates.shape[-1] == 3

        # Compute normalized camera coordinates
        normalized_coordinates = (
            self.K_inv @ coordinates[..., np.newaxis]
        )[..., 0]
        return dehomogenized(normalized_coordinates)  # (..., 2)

    def project(
        self,
        points: NDArray,
        return_depth: bool = False,
    ) -> NDArray | tuple[NDArray, NDArray]:
        """
        Project 3D world points to 2D pixel coordinates.

        Args:
            points: `NDArray(..., 4)` or `NDArray(..., 3)` homogeneous or
                    Euclidean points in the world frame

        Optional Args:
            return_depth: `bool` whether to also return the depth of the world
                          points. Default is `False`.

        Returns:
            pixels: `NDArray(..., 2)` projected 2D pixel coordinates
            depth: `NDArray(...)` depth of the world points. Returned only when
                   `return_depth` is `True`.
        """
        if points.shape[-1] == 3:
            points = homogenized(points)

        # Project world points
        pixels = dehomogenized((self.P @ points[..., None])[..., 0])

        # Compute depth if required
        if return_depth:
            points_camera = (
                self.pose.inverse.Rt @ points[..., None]
            )[..., 0]
            depth = points_camera[..., 2]  # (...,)
            return pixels, depth

        return pixels


    def backproject(self, coordinates: NDArray) -> NDArray:
        """
        Backproject pixel coordinates to the extrinsics frame.

        Args:
            coordinates: `NDArray(..., 3)` or `NDArray(..., 2)` homogeneous or
                         image coordinates. For 2D coordinates, the homogeneous
                         scale is assumed to be 1.

        Returns:
            points: `NDArray(..., 3)` points in the extrinsics frame
        """
        # Compute camera frame points
        points = self.backproject_camera_frame(coordinates)  # (..., 3)

        # Apply extrinsics
        if self.pose != Pose():
            # p_E = T_E_C * p_C
            points = self.pose * points
        return points

    def in_front_camera_frame(self, points: NDArray) -> NDArray:
        """
        Check whether camera-frame points are in front of the camera.

        Args:
            points: `NDArray(..., 3)` points in the camera frame

        Returns:
            in_front: `NDArray(...)` boolean mask indicating positive depth
        """
        # Check if depth is > 0
        return points[..., 2] > 0

    def in_front(self, points: NDArray) -> NDArray:
        """
        Check whether extrinsic-frame points are in front of the camera.

        Args:
            points: `NDArray(..., 3)` points in the extrinsics frame

        Returns:
            in_front: `NDArray(...)` boolean mask indicating positive depth
        """
        # Apply extrinsics
        points_camera = self.transform_to_camera_frame(points)
        return self.in_front_camera_frame(points_camera)

    def project_camera_frame_homogeneous(self, points: NDArray) -> NDArray:
        """
        Project 3D points from the camera frame to homogeneous image
        coordinates.

        Args:
            points: `NDArray(..., 3)` Euclidean points in the camera frame

        Returns:
            coordinates: `NDArray(..., 3)` homogeneous image coordinates
        """
        return (self.K @ points[..., np.newaxis])[..., 0]

    def project_camera_frame(self, points: NDArray) -> NDArray:
        """
        Project 3D points from the camera frame to image coordinates.

        Args:
            points: `NDArray(..., 3)` Euclidean points in the camera frame

        Returns:
            coordinates: `NDArray(..., 2)` image coordinates
        """
        return dehomogenized(self.project_camera_frame_homogeneous(points))

    def project_homogeneous(self, points: NDArray) -> NDArray:
        """
        Project 3D (homogeneous) points from the extrinsics frame to homogeneous
        image coordinates.

        Args:
            points: `NDArray(..., 3)` or `NDArray(..., 4)` Euclidean or
                    homogeneous points in the extrinsics frame

        Returns:
            homogeneous_coordinates: `NDArray(..., 3)` homogeneous image
                                     coordinates
        """
        # Make homogeneous
        if points.shape[-1] == 3:
            points = homogenized(points)
        assert points.shape[-1] == 4

        # Project
        return (self.P @ points[..., np.newaxis])[..., 0]

    def transform_to_camera_frame(self, points: NDArray) -> NDArray:
        """
        Transform 3D points from extrinsic frame to the camera frame.

        Args:
            points: `NDArray(..., 3)` points in the extrinsics frame

        Returns:
            points_camera: `NDArray(..., 3)` points in the camera frame
        """
        if self.pose != Pose():
            return self.pose.inverse * points
        return points

    def transform_to_extrinsic_frame(self, points: NDArray) -> NDArray:
        """
        Transform 3D points from the camera frame to the extrinsic frame.

        Args:
            points: `NDArray(..., 3)` points in the camera frame

        Returns:
            points_extrinsic: `NDArray(..., 3)` points in the extrinsics frame
        """
        if self.pose != Pose():
            return self.pose * points
        return self.pose * points
