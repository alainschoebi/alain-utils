# Pytest
import pytest

# Numpy
import numpy as np

# Utils
from alutils import homogenized, Pose

# ROS
try:
    import geometry_msgs.msg
    import std_msgs.msg
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

# COLMAP
try:
    import pycolmap
    COLMAP_AVAILABLE = True
except ImportError:
    COLMAP_AVAILABLE = False

# Logging
from alutils import get_logger
get_logger("alutils").setLevel("CRITICAL")

def test_general():

    for _ in range(100):
        q = np.random.randn(4)
        q /= np.linalg.norm(q)

        t = np.random.randn(3)

        pose = Pose.from_quat_wxyz(q, t)

        np.testing.assert_array_almost_equal(pose.R @ pose.R.T, np.eye(3))
        np.testing.assert_array_almost_equal(pose.R @ pose.inverse.R, np.eye(3))

        x = np.random.randn(3)
        np.testing.assert_array_almost_equal(
            x, pose.inverse.Rt @ homogenized((pose.Rt @ homogenized(x)))
        )
        np.testing.assert_array_almost_equal(x, pose.inverse * (pose * x))
        np.testing.assert_array_almost_equal(
            pose.quat_wxyz[[1, 2, 3, 0]], pose.quat_xyzw
        )
        np.testing.assert_array_almost_equal(
            pose.inverse.quat_wxyz[[1, 2, 3, 0]], pose.inverse.quat_xyzw
        )
        np.testing.assert_array_almost_equal(
            Pose.from_quat_wxyz(pose.quat_wxyz * np.array([-1, 1, 1, 1])).R,
            pose.inverse.R
        )
        np.testing.assert_array_almost_equal(
            Pose.from_quat_xyzw(pose.quat_xyzw * np.array([1, 1, 1, -1])).R,
            pose.inverse.R
        )

        pose = Pose.random()
        euler_xyz = pose.euler_angles('xyz')
        R_x = Pose.from_angle_and_axis(euler_xyz[0], [1, 0, 0]).R
        R_y = Pose.from_angle_and_axis(euler_xyz[1], [0, 1, 0]).R
        R_z = Pose.from_angle_and_axis(euler_xyz[2], [0, 0, 1]).R
        R = R_x @ R_y @ R_z
        np.testing.assert_array_almost_equal(pose.R, R)
        np.testing.assert_array_almost_equal(
            Pose.from_euler_angles(euler_xyz, 'xyz').R, pose.R
        )

        pose = Pose.random()
        euler_xyz = pose.euler_angles('zyz')
        R_0 = Pose.from_angle_and_axis(euler_xyz[0], [0, 0, 1]).R
        R_1 = Pose.from_angle_and_axis(euler_xyz[1], [0, 1, 0]).R
        R_2 = Pose.from_angle_and_axis(euler_xyz[2], [0, 0, 1]).R
        R = R_0 @ R_1 @ R_2
        np.testing.assert_array_almost_equal(pose.R, R)
        np.testing.assert_array_almost_equal(
            Pose.from_euler_angles(euler_xyz, 'zyz').R, pose.R
        )

def test_access_rights():

    p = Pose()

    with pytest.raises(Exception): p.R = 9
    with pytest.raises(Exception): p.R[0,0] = 9
    with pytest.raises(Exception): p.R[0,0] *= 9

    with pytest.raises(Exception): p.t = 9
    with pytest.raises(Exception): p.t[0] = 9
    with pytest.raises(Exception): p.t[0] *= 9

    R = p.R
    t = p.t

    with pytest.raises(Exception): R[0,0] = 9
    with pytest.raises(Exception): t[0] = 9

    p0 = Pose.from_angle_and_axis(1, np.array([1,1,1]))
    R = p0.R
    t = np.array([9, 8, 1])

    p1 = Pose(R, t)
    p2 = Pose(R)
    p2.t = t
    np.testing.assert_array_equal(p1.inverse.t, p2.inverse.t)

    p3 = Pose(np.eye(3), t)
    p4 = Pose(R, t)
    p3.R = R
    np.testing.assert_array_equal(p3.inverse.R, p4.inverse.R)
    np.testing.assert_array_equal(p3.inverse.quat_wxyz, p4.inverse.quat_wxyz)
    np.testing.assert_array_equal(p3.inverse.quat_xyzw, p4.inverse.quat_xyzw)


if ROS_AVAILABLE:
    def test_ros():
        for _ in range(100):
            pose = Pose.random()
            np.testing.assert_array_almost_equal(
                Pose.from_ros_pose(pose.to_ros_pose()).matrix, pose.matrix
            )

        assert 78 == 1

# TODO: to be fixed
if COLMAP_AVAILABLE and False:
    def test_colmap():
        for _ in range(100):
            pose = Pose.random()

            image = pycolmap.Image()
            Pose.set_colmap_image_pose(image, pose)
            np.testing.assert_array_almost_equal(
                pose.matrix, Pose.from_colmap_image(image).matrix
            )
            # pose: camera -> world
            # image: world -> camera
            np.testing.assert_array_almost_equal(
                image.qvec, pose.inverse.quat_wxyz
            )
