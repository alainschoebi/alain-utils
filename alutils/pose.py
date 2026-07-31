from __future__ import annotations

# Typing
from typing import Optional, overload

# NumPy
import numpy as np
from numpy.typing import NDArray

# ROS
try:
    import geometry_msgs.msg
    import std_msgs.msg
except ImportError:
    pass

# COLMAP
try:
    import pycolmap
except ImportError:
    pass

# Matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# SciPy
from scipy.spatial.transform import Rotation

# Utils
from .base import homogenized
from .decorators import requires_package

class Pose:
    """
    Pose class for 3D space.

    A `Pose` is defined by a 3x3 rotation matrix and a 3D translation vector.
    """

    # Default constructor
    def __init__(self, R: NDArray = np.eye(3),
                 t: NDArray | list = np.zeros(3),
                 tol: float = 1e-12) -> None:
        """
        Default constructor of the `Pose` class.

        Args:
            R: `NDArray(3,3)` 3D rotation matrix
            t: `NDArray(3,)` or `NDArray(3, 1)` or `list` 3D translation vector

        Optional Args:
            tol: `float` tolerance for validating the rotation matrix. Default
                 is 1e-12.
        """
        self.set_t(t)
        self.set_R(R, tol)

    # Setters
    def set_R(self, R: NDArray, tol: float = 1e-12):
        """
        Set the rotation matrix R.

        Args:
            R: `NDArray(3,3)` 3D rotation matrix

        Optional Args:
            tol: `float` tolerance for validating the rotation matrix. Default
                 is 1e-12.

        Requirements:
            3x3 real matrix
            orthogonal matrix, i.e. R^T @ R = I
            right handed, i.e. det(R) = +1
        """
        if not isinstance(R, np.ndarray):
            raise TypeError(
                f"The rotation matrix must be a NumPy array. Found `{type(R)}`."
            )
        if not R.shape == (3,3):
            raise ValueError("The rotation matrix must be a 3x3 matrix.")
        if not np.abs(np.linalg.det(R) - 1) < tol or \
           not np.all(np.abs((R.T @ R - np.eye(3))) < tol):
            raise ValueError(
                "The rotation matrix must be orthogonal and right-handed. Try "
                "reducing the `tol` parameter."
            )

        scipy_rotation = Rotation.from_matrix(R.copy())
        self.__R = scipy_rotation.as_matrix()
        self.__R.flags.writeable = False # Make read-only
        self.__inverse = Pose._compute_inverse(self)
        self.__quat_wxyz = scipy_rotation.as_quat()[[3, 0, 1, 2]]
        self.__quat_xyzw = self.__quat_wxyz[[1,2,3,0]]
        self.__quat_wxyz.flags.writeable = False # Make read-only
        self.__quat_xyzw.flags.writeable = False # Make read-only

    def set_t(self, t: NDArray | list):
        """
        Set the translation vector t.

        Args:
            t: `NDArray(3,)` or `NDArray(3, 1)` or `list` 3D translation vector
        """
        if not isinstance(t, (np.ndarray, list)):
            raise TypeError(
                f"Expected translation vector to be a 3D vector given as a "
                f"`NDArray(3,)`, `NDArray(3, 1)` or a `list`, found {type(t)}."
            )
        t = np.squeeze(np.array(t)).astype(float)
        if not t.shape == (3,):
            raise ValueError(
                f"Expected translation vector to be a 3D vector, but found "
                f"shape `{t.shape}`"
            )
        self.__t = t.copy() # t : 1D array (3,)
        self.__t.flags.writeable = False # Make read-only
        if hasattr(self, '_Pose__R'): # only if R is already set
            self.__inverse = Pose._compute_inverse(self)

    # Inverse computation
    @classmethod
    def _compute_inverse(cls, pose: Pose) -> Pose:
        """
        Class method that computes and sets the inverse of a `Pose`.

        Args:
            pose: `Pose` pose for which to compute the inverse

        Returns:
            inverse: `Pose` inverse of the given pose
        """
        # Create a new instance of the class without calling the constructor
        inv = cls.__new__(cls)
        inv.__R = pose.R.T
        inv.__t = -pose.R.T @ pose.t
        inv.__inverse = pose
        inv.__quat_wxyz = \
            Rotation.from_matrix(pose.R.T.copy()).as_quat()[[3, 0, 1, 2]]
        inv.__quat_xyzw = inv.__quat_wxyz[[1,2,3,0]]
        inv.__quat_wxyz.flags.writeable = False # Make read-only
        inv.__quat_xyzw.flags.writeable = False # Make read-only

        # Prohibit editing the attributes of the inverse Pose
        inv.set_R = cls._inverse_readonly_error
        inv.set_t = cls._inverse_readonly_error
        return inv


    # Inverse read-only error
    @staticmethod
    def _inverse_readonly_error(*args, **kwargs):
        """
        Raise an error when attempting to modify an inverse pose.

        Optional Args:
            *args: unused positional arguments
            **kwargs: unused keyword arguments

        Raises:
            AttributeError: inverse poses cannot be modified
        """
        raise AttributeError(
            "Can't set the rotation matrix or the translation vector of the " +
            "inverse of the Pose instance.")


    # Copy
    def copy(self) -> Pose:
        """
        Copy the `Pose` instance.

        Returns:
            pose: `Pose` copy of the instance
        """
        return Pose(self.R.copy(), self.t.copy())


    # Equality
    def __eq__(self, x):
        """
        Equality operator overload for two `Pose` instances.

        Args:
            x: object to compare with the `Pose` instance

        Returns:
            equal: `bool` whether the objects are equal
        """
        if not isinstance(x, Pose):
            return False

        return np.abs(self.matrix - x.matrix).max() < 1e-8


    # Properties and setters
    @property
    def R(self) -> NDArray:
        """
        Get the rotation matrix.

        Returns:
            R: `NDArray(3,3)` rotation matrix

        Note:
            This returns the actual reference to the .__R attribute. However,
            the .__R attribute is set as read-only, so it should not be
            possible to modify it.
        """
        return self.__R


    @R.setter
    def R(self, R: NDArray):
        """
        Set the rotation matrix R.

        Args:
            R: `NDArray(3,3)` 3D rotation matrix
        """
        self.set_R(R)


    @property
    def t(self) -> NDArray:
        """
        Get the translation vector.

        Returns:
            t: `NDArray(3,)` translation vector

        Note:
            This returns the actual reference to the .__t attribute. However,
            the .__t attribute is set as read-only, so it should not be
            possible to modify it.
        """
        return self.__t


    @t.setter
    def t(self, t: NDArray | list):
        """
        Set the translation vector t.

        Args:
            t: `NDArray(3,)` or `NDArray(3, 1)` or `list` 3D translation vector
        """
        self.set_t(t)


    @property
    def Rt(self) -> NDArray:
        """
        Get the homogeneous transformation matrix without its last row.

        Returns:
            Rt: `NDArray(3,4)` homogeneous transformation matrix
        """
        return np.c_[self.R, self.t]


    @property
    def matrix(self) -> NDArray:
        """
        Get the homogeneous transformation matrix.

        Returns:
            matrix: `NDArray(4,4)` homogeneous transformation matrix
        """
        return np.r_[self.Rt, np.array([[0, 0, 0, 1]])]


    @property
    def inverse(self) -> Pose:
        """
        Get the inverse of the `Pose` instance.

        Returns:
            inverse: `Pose` inverse of the instance

        Note:
            This returns the actual reference to the .__inverse attribute.
            However, the `set_R` and `set_t` methods are unavailable for the
            inverse pose, so it should not be possible to modify it.
        """
        return self.__inverse


    @property
    def quat_xyzw(self) -> NDArray:
        """
        Get the quaternion representation of the rotation as an `NDArray(4, )`.

        Returns:
            quaternion: `NDArray(4,)` quaternion in `(x, y, z, w)` order

        Note:
            This returns the actual reference to the .__quat_xyzw attribute.
            However, the .__quat_xyzw attribute is set as read-only, so it
            should not be possible to modify it.
        """
        return self.__quat_xyzw


    @property
    def quat_wxyz(self) -> NDArray:
        """
        Get the quaternion representation of the rotation as an `NDArray(4, )`.

        Returns:
            quaternion: `NDArray(4,)` quaternion in `(w, x, y, z)` order

        Note:
            This returns the actual reference to the .__quat_wxyz attribute.
            However, the .__quat_wxyz attribute is set as read-only, so it
            should not be possible to modify it.
        """
        return self.__quat_wxyz


    # Methods
    def angle_axis(self) -> NDArray:
        """
        Get the 3D angle-axis vector representing the rotation matrix.

        The direction of the vector is the axis of rotation and its norm is the
        angle of rotation in radians.

        Returns:
            angle_axis: `NDArray(3,)` the angle-axis 3D vector
        """
        return Rotation.from_matrix(self.R.copy()).as_rotvec()


    def rotation_angle_and_axis(self) -> tuple[float, NDArray]:
        """
        Compute the angle and axis of the rotation matrix of the pose. The angle
        is given in radians and the axis is a unit 3D vector.

        Returns:
            angle: `float` the angle in radians
            axis: `NDArray(3,)` the axis of the rotation (unit vector)
        """
        angle_axis = self.angle_axis()
        angle = float(np.linalg.norm(angle_axis))
        axis = angle_axis / angle
        return angle, axis

    def euler_angles(self, seq: str, degrees: Optional[bool] = False) \
        -> NDArray:

        """
        Returns the Euler angles from the rotation matrix of the pose. Any
        rotation can be represented by three consecutive rotations about the
        axes of a coordinate system. The Euler angles are the angles of these
        rotations.

        Due to Gimbal lock issue, the Euler angles are not always unique. In
        this case, the function will trigger a warning. TODO???

        Example:
            Given a rotation matrix R, and the sequence 'xyz', the function will
            return the Euler angles [alpha, beta, gamma] such that:
                R = R_x(alpha) @ R_y(beta) @ R_z(gamma)
            where R_x(alpha) represents a rotation of angle alpha about the
            x-axis.

        Args:
            seq: `str` the sequence of axes of rotation. The string must contain
                 exactly 3 characters belonging to the set 'x', 'y', 'z'.
                 Adjacent axes cannot be the same.

        Optional Args:
            degrees: `bool` if `True`, return the Euler angles in degrees;
                     otherwise, return them in radians. Default is radians..

        Returns:
            angles: `NDArray(3,)` the Euler angles in radians (or degrees)

        Note:
            The SciPy library distinguishes between internal and external
            rotations.
        """

        if not len(seq) == 3 or not all([c in 'xyz' for c in seq]) or \
            seq[0] == seq[1] or seq[1] == seq[2]:
            raise ValueError(
                "The sequence of axes must be a string of three characters " +
                "belonging to the set {'x', 'y', 'z'}. It can't same " +
                "adjacent axes."
            )

        return Rotation.from_matrix(self.R.copy()) \
               .as_euler(seq.upper(), degrees=degrees)


    # Pose errors
    @staticmethod
    def distance_error(p1: Pose, p2: Pose) -> float:
        """
        Compute the distance error between two `Pose` instances. It computes the
        norm of the difference of the two translations vectors.

        Args:
            p1: `Pose` first pose
            p2: `Pose` second pose

        Returns:
            error: `float` the distance error
        """
        return float(np.linalg.norm(p1.t - p2.t))


    @staticmethod
    def angular_error(p1: Pose, p2: Pose,
                      degrees: Optional[bool] = False) -> float:
        """
        Compute the angular error between two `Pose` instances. It computes the
        angle between the two rotation matrices. The angular error is given in
        radians unless the `degrees` flag is set to True.

        Args:
            p1: `Pose` first pose
            p2: `Pose` second pose

        Optional Args:
            degrees: `bool` if `True`, the angular error is returned in degrees,
                     otherwise in radians. Default is radians.

        Returns:
            error: `float` the angular error in radians (or degrees)
        """
        pose_diff = p1 * p2.inverse
        angle = float(np.linalg.norm(pose_diff.angle_axis()))
        return angle if not degrees else np.rad2deg(angle)


    @staticmethod
    def error(p1: Pose, p2: Pose,
              degrees: Optional[bool] = False) -> tuple[float, float]:
        """
        Compute the distance and angular error between two `Pose` instances.

        Args:
            p1: `Pose` first pose
            p2: `Pose` second pose

        Optional Args:
            degrees: `bool` if `True`, the angular error is returned in degrees,
                     otherwise in radians. Default is radians.

        Returns:
            distance_error: `float` the distance error
            angular_error: `float` the angular error in radians (or degrees)
        """
        return Pose.distance_error(p1, p2), \
               Pose.angular_error(p1, p2, degrees=degrees)


    # Additional constructors
    @staticmethod
    def random(t_bound: float = 1) -> Pose:
        """
        Generates a random `Pose` with any orientation and position in the
        bounded cube defined by [-t_bound, +t_bound]^3.

        Optional Args:
            t_bound: `float` the bound for the translation vector. Default is 1.

        Returns:
            pose: `Pose` random pose
        """
        t = (np.random.random(3) - 0.5) * 2 * t_bound
        zyx = np.random.random(3) * 2 * np.pi # radians
        r = Rotation.from_euler('zyx', zyx)
        return Pose(r.as_matrix(), t)


    @staticmethod
    def from_angle_and_axis(
            angle: float,
            axis: NDArray | list,
            t: NDArray | list = np.zeros(3),
            degrees: Optional[bool] = False
        ) -> Pose:
        """
        Get a `Pose` from an angle in radians (or degrees) and a 3D axis vector.

        Args:
            angle: `float` the angle of rotation in radians (or degrees)
            axis: `NDArray(3,)` or `NDArray(3, 1)` or `list` the axis of
                  rotation. The axis vector does not necessarily need to be
                  normalized.

        Optional Args:
            t: `NDArray(3,)`, `NDArray(3, 1)`, or `list` the 3D translation
               vector. Default is the zero vector.
            degrees: `bool` if `True`, the angle is given in degrees, otherwise
                     in radians. Default is radians.

        Returns:
            pose: `Pose` pose created from the angle, axis, and translation
        """
        axis = np.squeeze(np.array(axis)).astype(float)
        assert axis.shape == (3,)
        axis = axis / np.linalg.norm(axis)
        if degrees: angle = np.deg2rad(angle)
        rot_vec = axis * angle
        return Pose.from_rotation_vector(rot_vec, t)


    @staticmethod
    def from_rotation_vector(rot_vec: NDArray | list,
                             t: NDArray | list = np.zeros(3)) -> Pose:
        """
        Get a `Pose` from a rotation vector that captures the axis and the angle
        through its norm.

        Args:
            rot_vec: `NDArray(3,)` or `NDArray(3, 1)` or `list` the input
                     rotation vector

        Optional Args:
            t: `NDArray(3,)`, `NDArray(3, 1)`, or `list` the 3D translation
               vector. Default is the zero vector.

        Returns:
            pose: `Pose` pose created from the rotation vector and translation
        """
        rot_vec = np.squeeze(np.array(rot_vec)).astype(float)
        assert rot_vec.shape == (3,)
        R = Rotation.from_rotvec(rot_vec).as_matrix()
        return Pose(R, t)


    @staticmethod
    def from_quat_xyzw(q: NDArray | list,
                       t: NDArray | list = np.zeros(3)) -> Pose:
        """
        Get a `Pose` from quaternions `q = (x, y, z, w)` and a translation
        vector `t = (x, y, z)`.

        Args:
            q: `NDArray(4,)` or `list` the input quaternion `(x, y, z, w)`

        Optional Args:
            t: `NDArray(3,)`, `NDArray(3, 1)`, or `list` the 3D translation
               vector. Default is the zero vector.

        Returns:
            pose: `Pose` pose created from the quaternion and translation

        Note:
            The quaternion does not necessarily need to be normalized.
        """
        q = np.array(q).squeeze()
        assert q.shape == (4,)
        R = Rotation.from_quat(q).as_matrix() # assumes (x, y, z, w) quaternions
        return Pose(R, t)


    @staticmethod
    def from_quat_wxyz(q: NDArray | list,
                       t: NDArray | list = np.zeros(3)) -> Pose:
        """
        Get a `Pose` from quaternions `q = (w, x, y, z)` and a translation
        vector `t = (x, y, z)`.

        Args:
            q: `NDArray(4,)` or `list` the input quaternion `(w, x, y, z)`

        Optional Args:
            t: `NDArray(3,)`, `NDArray(3, 1)`, or `list` the 3D translation
               vector. Default is the zero vector.

        Returns:
            pose: `Pose` pose created from the quaternion and translation

        Note:
            The quaternion does not necessarily need to be normalized.
        """
        return Pose.from_quat_xyzw([q[1], q[2], q[3], q[0]], t)


    @staticmethod
    def from_matrix(matrix: NDArray, tol: float = 1e-12) -> Pose:
        """
        Get a `Pose` from a 3x4 or 4x4 homogeneous transformation matrix.

        Args:
            matrix: `NDArray(3,4)` or `NDArray(4,4)` the homogeneous
                    transformation matrix

        Optional Args:
            tol: `float` tolerance for validating the rotation matrix. Default
                 is 1e-12.

        Returns:
            pose: `Pose` pose created from the transformation matrix
        """
        assert (matrix.shape == (4,4) or matrix.shape == (3,4)) and \
               "Not a 3x4 or 4x4 homogeneous matrix"
        return Pose(matrix[:3, :3], matrix[:3, 3], tol=tol)


    @staticmethod
    def from_euler_angles(angles: tuple[float] | list[float] | NDArray,
                          seq: str, degrees: Optional[bool] = False,
                          t: NDArray | list[float] = np.zeros(3)) -> Pose:

        """
        Get a `Pose` from 3 Euler angles.

        Example:
            Given 3 Euler angles [alpha, beta, gamma], and the sequence 'xyz',
            the function will return the Pose with rotation matrix given by:
                R = R_x(alpha) @ R_y(beta) @ R_z(gamma)
            where R_x(alpha) represents a rotation of angle alpha about the
            x-axis.

        Args:
            angles: `NDArray(3,)`, `tuple` or `list` the Euler angles in radians
                    (or degrees)
            seq: `str` the sequence of axes of rotation. The string must
                 contain exactly 3 characters belonging to the set 'x', 'y',
                 'z'. Adjacent axes cannot be the same.

        Optional Args:
            degrees: `bool` if `True`, the angles are given in degrees;
                     otherwise, they are given in radians. Default is `False`.
            t: `NDArray(3,)`, `NDArray(3, 1)`, or `list` the 3D translation
               vector. Default is the zero vector.

        Returns:
            pose: `Pose` pose created from the Euler angles and translation
        """

        if not len(seq) == 3 or not all([c in 'xyz' for c in seq]) or \
            seq[0] == seq[1] or seq[1] == seq[2]:
            raise ValueError(
                "The sequence of axes must be a string of three characters " +
                "belonging to the set {'x', 'y', 'z'}. It can't same " +
                "adjacent axes."
            )

        angles = np.array(angles)
        assert len(angles) == 3

        rotation = Rotation.from_euler(seq.upper(), angles, degrees=degrees)
        return Pose(rotation.as_matrix(), t)


    # ROS constructors and methods
    @staticmethod
    @requires_package('rospy')
    def from_ros_pose(pose_msg: geometry_msgs.msg.Pose) -> Pose:
        """
        Get a `Pose` from a ROS pose.

        Args:
            pose_msg: `geometry_msgs.msg.Pose` ROS pose message

        Returns:
            pose: `Pose` pose created from the ROS pose message
        """
        q = np.array([pose_msg.orientation.w, pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z])
        t = np.array([pose_msg.position.x, pose_msg.position.y, pose_msg.position.z])
        return Pose.from_quat_wxyz(q, t)


    @staticmethod
    @requires_package('rospy')
    def from_ros_transform(transform_msg: geometry_msgs.msg.Transform) -> Pose:
        """
        Get a `Pose` from a ROS transform.

        Args:
            transform_msg: `geometry_msgs.msg.Transform` ROS transform message

        Returns:
            pose: `Pose` pose created from the ROS transform message
        """
        q = np.array([transform_msg.rotation.w, transform_msg.rotation.x, transform_msg.rotation.y, transform_msg.rotation.z])
        t = np.array([transform_msg.translation.x, transform_msg.translation.y, transform_msg.translation.z])
        return Pose.from_quat_wxyz(q, t)


    @requires_package('rospy')
    def to_ros_pose(self) -> geometry_msgs.msg.Pose:
        """
        Transform the `Pose` into a ROS pose.

        Returns:
            pose_msg: `geometry_msgs.msg.Pose` ROS pose message
        """
        return geometry_msgs.msg.Pose(position=geometry_msgs.msg.Point(*self.t),
                                    orientation=geometry_msgs.msg.Quaternion(*self.quat_xyzw))


    @requires_package('rospy')
    def to_ros_pose_stamped(self, header: std_msgs.msg.Header) -> geometry_msgs.msg.PoseStamped:
        """
        Transform the `Pose` into a stamped ROS pose.

        Args:
            header: `std_msgs.msg.Header` ROS message header

        Returns:
            pose_msg: `geometry_msgs.msg.PoseStamped` stamped ROS pose message
        """
        return geometry_msgs.msg.PoseStamped(header, self.to_ros_pose())


    @requires_package('rospy')
    def to_ros_transform(self) -> geometry_msgs.msg.Transform:
        """
        Transform the `Pose` into a ROS transform.

        Returns:
            transform_msg: `geometry_msgs.msg.Transform` ROS transform message
        """
        return geometry_msgs.msg.Transform(translation=geometry_msgs.msg.Vector3(*self.t),
                                        rotation=geometry_msgs.msg.Quaternion(*self.quat_xyzw))


    @requires_package('rospy')
    def to_ros_transform_stamped(self, header: std_msgs.msg.Header, child_frame_id: str) \
        -> geometry_msgs.msg.TransformStamped:
        """
        Transform the `Pose` into a stamped ROS transform.

        Args:
            header: `std_msgs.msg.Header` ROS message header
            child_frame_id: `str` name of the child coordinate frame

        Returns:
            transform_msg: `geometry_msgs.msg.TransformStamped` stamped ROS
                           transform message
        """
        return geometry_msgs.msg.TransformStamped(header, child_frame_id, self.to_ros_transform())


    # COLMAP constructors and methods
    @staticmethod
    @requires_package('pycolmap')
    def from_colmap_image(image: pycolmap.Image,
                          include_name: bool = True) -> Pose:
        """
        Get a `Pose` from a COLMAP image.

        Args:
            image: `pycolmap.Image` COLMAP image

        Optional Args:
            include_name: `bool` whether to attach the image name to the pose.
                          Default is `True`.

        Returns:
            pose: `Pose` pose created from the COLMAP image

        Note:
            It returns the transformation from camera to world, that is,
            `t = (tx, ty, tz)` contains the coordinates of the camera in the
            world.
        """
        return NotImplementedError("Need to be fixed.")
        pose = Pose.from_quat_wxyz(image.qvec, image.tvec).inverse
        if include_name: pose.name = image.name
        return pose


    @staticmethod
    @requires_package('pycolmap')
    def set_colmap_image_pose(image: pycolmap.Image, pose:Pose) -> None:
        """
        Sets the pose of the COLMAP image to the given `Pose`.

        Args:
            image: `pycolmap.Image` COLMAP image to update
            pose: `Pose` pose to assign to the COLMAP image

        Note:
            The given `Pose` must be the transformation from camera to world,
            that is, `t = (tx, ty, tz)` contains the coordinates of the camera
            in the world.
        """
        return NotImplementedError("Need to be fixed.")
        q = pose.inverse.quat_wxyz
        t = pose.inverse.t
        image.qvec, image.tvec = q, t
        return


    @overload
    def __mul__(self, x: NDArray) -> NDArray: ...

    @overload
    def __mul__(self, x: Pose) -> Pose: ...

    # Multiplication operator overload
    def __mul__(self, x: NDArray | Pose) -> NDArray | Pose:
        """
        Multiplication `*` operator overload.

        Args:
            x: `NDArray(..., 3)`, `NDArray(..., 3, 1)`, or `Pose` value to
               multiply with this pose

        Returns:
            result: `NDArray(..., 3)`, `NDArray(..., 3, 1)`, or `Pose`
                    transformed vectors or composed pose

        Example:
            With an `NDArray`, `y = pose * x` applies the transformation
            described by `pose` to the 3D vectors in `x`.

            With another `Pose`, `pose_A_C = pose_A_B * pose_B_C` represents
            the consecutive transformations described by `pose_A_B` and
            `pose_B_C`.
        """

        # With a NDArray: y = pose * x
        if isinstance(x, np.ndarray):
            # x: (..., 3) or (..., 3, 1)
            if x.shape[-1] == 1:
                x = x[..., 0]
            assert x.shape[-1] == 3
            return (self.Rt @ homogenized(x)[..., np.newaxis])[..., 0]

        # With another Pose: pose_A_C = pose_A_B * pose_B_C.
        elif isinstance(x, Pose):
            pose_matrix = self.matrix @ x.matrix # (4,4)
            return Pose(pose_matrix[:3, :3], pose_matrix[:3, 3])

        # Undefined multiplication
        else:
            raise NotImplementedError(
                f"Multiplication of a Pose with {type(x)} is not defined."
            )


    @staticmethod
    def interpolate(p1: Pose, p2: Pose, alpha: float) -> Pose:
        """
        Interpolate between two `Pose` instances using SLERP.

        Args:
            p1: `Pose` first pose
            p2: `Pose` second pose
            alpha: `float` interpolation factor in the range [0, 1]

        Returns:
            pose: `Pose` the interpolated pose between p1 and p2
        """

        assert 0 <= alpha <= 1 and "Alpha must be in the range [0, 1]"

        # Interpolate translation
        t = (1 - alpha) * p1.t + alpha * p2.t

        # Interpolate rotation
        R_12 = p1.R.T @ p2.R
        angle_12, axis_12 = Pose(R_12).rotation_angle_and_axis()
        angle = alpha * angle_12
        R = p1.R @ Pose.from_angle_and_axis(angle, axis_12).R
        return Pose(R, t)


    # Representations
    def __repr__(self):
        return self.__str__()


    def __str__(self):
        return "Pose with rotation R:\n" + str(np.round(self.__R,2)) + \
               "\ntranslation t:\n " + str(np.round(self.__t,2))


    # Visualization functions
    def show(self, **args) -> Axes3D:
        """
        Visualize the `Pose` in a 3D matplotlib plot.

        Optional Args:
            **args: optional arguments forwarded to `Pose.visualize`

        Returns:
            axes: `Axes3D` matplotlib axes containing the pose visualization
        """
        return Pose.visualize(self, **args)


    @staticmethod
    def visualize(poses: Pose | list[Pose],
                  axes: Optional[Axes3D] = None,
                  savefig: Optional[str] = None,
                  show: Optional[bool] = True,
                  scale: Optional[float] = None,
                  ) -> Axes3D:
        """
        Visualize a `Pose` or a list of poses in a 3D matplotlib plot.

        Args:
            poses: `Pose` or `list[Pose]` pose(s) to plot

        Optional Args:
            axes: `Axes` matplotlib axes to plot on. If not provided, a new
                  figure will be created.
            savefig: `str` path to save the figure. By default the figure is not
                     being saved.
            show: `bool` whether to show the plot or not. Default is True. If
                  axes are provided, the plot will not be shown. If savefig is
                  provided, the plot will not be shown.
            scale: `float` length scale for the plotted coordinate axes. If not
                   provided, it is determined from the poses' positions.

        Returns:
            axes: `Axes3D` matplotlib axes containing the pose visualization

        """

        # Assertions
        if savefig and axes:
            raise ValueError("Can't save figure if axes are provided.")

        # Do not show
        if savefig or axes:
            show = False

        if isinstance(poses, Pose):
            poses = [poses]

        # No axes provided
        if axes is None:
            # Create figure
            fig = plt.figure()
            ax: Axes3D = fig.add_subplot(111, projection='3d')

            # Title
            ax.set_title(f"Pose{'s' if len(poses) > 1 else ''} visualization")

            # Axis labels
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')

        # Axes provided
        else:
            ax = axes

        # Access poses origins
        origins = np.array([pose.t for pose in poses]) # (N, 3)
        origins_max = np.max(origins, axis = 0)
        origins_min = np.min(origins, axis = 0)
        max_diff = np.max(origins_max - origins_min)
        max_diff = max(max_diff, 0.1)

        # Plot poses
        for i, pose in enumerate(poses):

            # Origin
            ax.scatter(*list(pose.t), c='k', marker='o')

            # Direction vectors
            s = scale if scale is not None else max_diff/15 # scale of the arrows

            params = {'arrow_length_ratio': 0.3, 'linewidth': 2}
            Pose._plot_arrow(ax, pose.t, s * pose.R[:, 0], color='r', **params)
            Pose._plot_arrow(ax, pose.t, s * pose.R[:, 1], color='g', **params)
            Pose._plot_arrow(ax, pose.t, s * pose.R[:, 2], color='b', **params)

            # Text
            if not (len(poses) == 1 and not hasattr(pose, 'name')):
                text = i if not hasattr(pose, 'name') else pose.name
                ax.text(*list(pose.t), text, color='k', fontsize=12)

        if axes is None:
            # Equal axis
            ax.axis('equal')
            ax.set_box_aspect([1, 1, 1])

        # Show
        if show: plt.show()

        # Save figure
        if savefig: fig.savefig(savefig)

        if not show and axes is None:
            plt.close(fig)

        return ax


    @staticmethod
    def _plot_arrow(ax: Axes3D, origin: NDArray, vector: NDArray, **args) -> None:
        """
        Plot a 3D arrow.

        Args:
            ax: `Axes3D` matplotlib axes on which to plot the arrow
            origin: `NDArray(3,)` origin of the arrow
            vector: `NDArray(3,)` direction and length of the arrow

        Optional Args:
            **args: optional arguments forwarded to `Axes3D.quiver`
        """
        ax.plot3D([origin[0], origin[0] + vector[0]],
                    [origin[1], origin[1] + vector[1]],
                    [origin[2], origin[2] + vector[2]],
                    color="k")

        # Plot the arrows
        ax.quiver(origin[0], origin[1], origin[2],
                    vector[0], vector[1], vector[2],
                    **args)
        return
