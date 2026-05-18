"""Unit tests for transform.py module."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from neuracore_types.importer.config import (
    XYZ,
    AngleConfig,
    EulerOrderConfig,
    Frame,
    FrameTransformConfig,
    ImageChannelOrderConfig,
    ImageConventionConfig,
    PoseConfig,
    QuaternionOrderConfig,
    RollPitchYaw,
    RotationConfig,
)
from neuracore_types.importer.transform import (
    ApplyFrameTransform,
    CastToNumpyDtype,
    Clip,
    DataTransformSequence,
    DegreesToRadians,
    FlipSign,
    ImageChannelOrder,
    ImageFormat,
    LanguageFromBytes,
    NanToNum,
    Normalize,
    NumpyToScalar,
    Offset,
    Pose,
    Rotation,
    Scale,
    Unnormalize,
)


class TestDataTransformSequence:
    """Tests for DataTransformSequence class."""

    def test_data_transform_sequence_empty(self):
        """Test DataTransformSequence with empty transforms."""
        sequence = DataTransformSequence(transforms=[])
        data = np.array([1, 2, 3])
        result = sequence(data)
        np.testing.assert_array_equal(result, data)

    def test_data_transform_sequence_multiple_transforms(self):
        """Test DataTransformSequence with multiple transforms."""
        sequence = DataTransformSequence(
            transforms=[Scale(factor=2.0), Offset(value=1.0)]
        )
        data = np.array([1.0, 2.0, 3.0])
        result = sequence(data)
        expected = np.array([3.0, 5.0, 7.0])  # (x * 2) + 1
        np.testing.assert_array_equal(result, expected)


class TestRotation:
    """Tests for Rotation transform."""

    def test_rotation_quaternion_xyzw(self):
        """Test Rotation with quaternion xyzw (no change)."""
        transform = Rotation(
            rotation_type=RotationConfig.QUATERNION,
            angle_type=AngleConfig.RADIANS,
            seq=QuaternionOrderConfig.XYZW,
        )
        quat = np.array([0.5, 0.5, 0.5, 0.5])
        result = transform(quat)
        np.testing.assert_array_almost_equal(result, quat)

    def test_rotation_quaternion_wxyz(self):
        """Test Rotation with quaternion wxyz."""
        transform = Rotation(
            rotation_type=RotationConfig.QUATERNION,
            angle_type=AngleConfig.RADIANS,
            seq=QuaternionOrderConfig.WXYZ,
        )
        quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
        result = transform(quat_wxyz)
        expected = np.array([0.0, 0.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_rotation_matrix(self):
        """Test Rotation with rotation matrix."""
        transform = Rotation(
            rotation_type=RotationConfig.MATRIX, angle_type=AngleConfig.RADIANS
        )
        # Identity rotation matrix
        rot_matrix = np.eye(3)
        result = transform(rot_matrix)
        # Identity matrix should give quaternion [0, 0, 0, 1] (xyzw)
        expected = np.array([0.0, 0.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_rotation_euler_radians(self):
        """Test Rotation with euler angles in radians."""
        transform = Rotation(
            rotation_type=RotationConfig.EULER,
            angle_type=AngleConfig.RADIANS,
            seq=EulerOrderConfig.XYZ.value,
        )
        euler = np.array([0.0, 0.0, 0.0])
        result = transform(euler)
        # Zero euler should give quaternion [0, 0, 0, 1]
        expected = np.array([0.0, 0.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_apply_frame_transform_body_rotation_only(self):
        """BODY: T' = T @ X (post-multiply).

        For an identity input orientation a body rotation rotates the
        orientation about its local axes and leaves position unchanged. Used
        when the dataset's tool identity is rotated from the URDF link's
        identity (e.g. bridge: gripper-down vs gripper-forward).
        """
        pose = np.array([0.3, 0.05, 0.07, 0.0, 0.0, 0.0, 1.0])
        transform = ApplyFrameTransform(
            transforms=[
                FrameTransformConfig(
                    frame=Frame.TOOL,
                    rotation=RollPitchYaw(pitch=np.pi / 2),
                )
            ]
        )
        out = transform(pose)
        np.testing.assert_array_almost_equal(out[:3], pose[:3])
        np.testing.assert_array_almost_equal(
            out[3:7], R.from_euler("y", np.pi / 2).as_quat()
        )

    def test_apply_frame_transform_world_rotation_pre_multiplies(self):
        """WORLD: T' = X @ T (pre-multiply) -- re-base into a rotated frame."""
        pose = np.array([0.3, 0.05, 0.07, 0.0, 0.0, 0.0, 1.0])
        transform = ApplyFrameTransform(
            transforms=[
                FrameTransformConfig(
                    frame=Frame.WORLD,
                    rotation=RollPitchYaw(pitch=np.pi / 2),
                )
            ]
        )
        out = transform(pose)
        # R_y(pi/2) @ [0.3, 0.05, 0.07] = [0.07, 0.05, -0.3]
        np.testing.assert_array_almost_equal(out[:3], [0.07, 0.05, -0.3])
        np.testing.assert_array_almost_equal(
            out[3:7], R.from_euler("y", np.pi / 2).as_quat()
        )

    def test_apply_frame_transform_world_then_body_is_conjugation(self):
        """WORLD(X) + BODY(X^-1) reproduces a conjugation X @ T @ X^-1.

        This is how the former FRAME mode (re-framing an action delta) is now
        expressed, and must hold for a fully general pose and rotation.
        """
        pose = np.array([0.3, 0.05, 0.07, 0.1, 0.2, 0.3, 0.9])
        pose[3:7] /= np.linalg.norm(pose[3:7])
        rpy = [0.4, -0.6, 1.1]
        X = np.eye(4)
        X[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
        T = np.eye(4)
        T[:3, 3] = pose[:3]
        T[:3, :3] = R.from_quat(pose[3:7]).as_matrix()
        expected = X @ T @ np.linalg.inv(X)
        inv_rpy = list(R.from_euler("xyz", rpy).inv().as_euler("xyz"))
        transform = ApplyFrameTransform(
            transforms=[
                FrameTransformConfig(
                    frame=Frame.WORLD,
                    rotation=RollPitchYaw(roll=rpy[0], pitch=rpy[1], yaw=rpy[2]),
                ),
                FrameTransformConfig(
                    frame=Frame.TOOL,
                    rotation=RollPitchYaw(
                        roll=inv_rpy[0], pitch=inv_rpy[1], yaw=inv_rpy[2]
                    ),
                ),
            ]
        )
        out = transform(pose)
        np.testing.assert_array_almost_equal(out[:3], expected[:3, 3])
        np.testing.assert_array_almost_equal(
            out[3:7], R.from_matrix(expected[:3, :3]).as_quat()
        )

    def test_apply_frame_transform_body_translation_in_tool_frame(self):
        """BODY translation offsets position along the body's own axes."""
        # Pose rotated +90 deg about z; a +x body offset moves +y in the world.
        quat = R.from_euler("z", np.pi / 2).as_quat()
        pose = np.concatenate([[1.0, 2.0, 3.0], quat])
        transform = ApplyFrameTransform(
            transforms=[FrameTransformConfig(frame=Frame.TOOL, translation=XYZ(x=0.5))]
        )
        out = transform(pose)
        np.testing.assert_array_almost_equal(out[:3], [1.0, 2.5, 3.0])
        np.testing.assert_array_almost_equal(out[3:7], quat)

    def test_apply_frame_transform_world_translation_in_reference_frame(self):
        """WORLD translation offsets position along the reference-frame axes."""
        pose = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])
        transform = ApplyFrameTransform(
            transforms=[FrameTransformConfig(frame=Frame.WORLD, translation=XYZ(x=0.5))]
        )
        out = transform(pose)
        np.testing.assert_array_almost_equal(out[:3], [1.5, 2.0, 3.0])
        np.testing.assert_array_almost_equal(out[3:7], pose[3:7])

    def test_apply_frame_transform_empty_is_identity(self):
        """An empty transform list leaves the pose unchanged."""
        pose = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.9])
        pose[3:7] /= np.linalg.norm(pose[3:7])
        out = ApplyFrameTransform(transforms=[])(pose)
        np.testing.assert_array_almost_equal(out[:3], pose[:3])
        # Compare as matrices to stay invariant to quaternion sign.
        np.testing.assert_array_almost_equal(
            R.from_quat(out[3:7]).as_matrix(), R.from_quat(pose[3:7]).as_matrix()
        )

    def test_apply_frame_transform_world_full_se3(self):
        """WORLD with rotation AND translation: T' = X @ T, X = [R | t]."""
        pose = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        transform = ApplyFrameTransform(
            transforms=[
                FrameTransformConfig(
                    frame=Frame.WORLD,
                    rotation=RollPitchYaw(yaw=np.pi / 2),
                    translation=XYZ(x=1.0, y=2.0, z=3.0),
                )
            ]
        )
        out = transform(pose)
        # R_z(90) @ [1,0,0] + [1,2,3] = [0,1,0] + [1,2,3] = [1,3,3].
        np.testing.assert_array_almost_equal(out[:3], [1.0, 3.0, 3.0])
        np.testing.assert_array_almost_equal(
            out[3:7], R.from_euler("z", np.pi / 2).as_quat()
        )

    def test_apply_frame_transform_body_full_se3(self):
        """BODY with rotation AND translation: T' = T @ X, X = [R | t].

        The body translation is expressed in the pose's own frame, so it is
        rotated by the pose orientation before being added.
        """
        quat = R.from_euler("z", np.pi / 2).as_quat()
        pose = np.concatenate([[1.0, 2.0, 3.0], quat])
        transform = ApplyFrameTransform(
            transforms=[
                FrameTransformConfig(
                    frame=Frame.TOOL,
                    rotation=RollPitchYaw(yaw=np.pi / 2),
                    translation=XYZ(x=1.0),
                )
            ]
        )
        out = transform(pose)
        # pos: R_z(90) @ [1,0,0] + [1,2,3] = [1,3,3]; orient: R_z(90) @ R_z(90).
        np.testing.assert_array_almost_equal(out[:3], [1.0, 3.0, 3.0])
        np.testing.assert_array_almost_equal(
            out[3:7], R.from_euler("z", np.pi).as_quat()
        )

    def test_rotation_euler_extrinsic_xyz_matches_ros_static_axes(self):
        """extrinsic_euler=True must produce extrinsic (ROS static-axes) quaternions.

        For combined rotations, intrinsic XYZ and extrinsic XYZ diverge. This
        verifies the opt-in flag actually switches conventions.
        """
        euler = np.array([0.5, -0.7, 1.2])  # well outside the small-angle regime
        intrinsic = Rotation(
            rotation_type=RotationConfig.EULER,
            angle_type=AngleConfig.RADIANS,
            seq=EulerOrderConfig.XYZ,
        )
        extrinsic = Rotation(
            rotation_type=RotationConfig.EULER,
            angle_type=AngleConfig.RADIANS,
            seq=EulerOrderConfig.XYZ,
            extrinsic_euler=True,
        )
        q_int = intrinsic(euler)
        q_ext = extrinsic(euler)
        # Intrinsic XYZ (current default) must still equal scipy lowercase 'xyz'.
        np.testing.assert_array_almost_equal(
            q_int, R.from_euler("xyz", euler).as_quat()
        )
        # Extrinsic XYZ (opt-in) must equal scipy uppercase 'XYZ'.
        np.testing.assert_array_almost_equal(
            q_ext, R.from_euler("XYZ", euler).as_quat()
        )
        # The two conventions must actually differ for these angles.
        assert not np.allclose(q_int, q_ext)

    def test_rotation_euler_degrees(self):
        """Test Rotation with euler angles in degrees."""
        transform = Rotation(
            rotation_type=RotationConfig.EULER,
            angle_type=AngleConfig.DEGREES,
            seq=EulerOrderConfig.XYZ,
        )
        euler = np.array([90.0, 0.0, 0.0])
        result = transform(euler)
        # 90 degree rotation around x-axis
        expected = R.from_euler("xyz", [90.0, 0.0, 0.0], degrees=True).as_quat()
        np.testing.assert_array_almost_equal(result, expected)

    def test_rotation_axis_angle_radians(self):
        """Test Rotation with axis-angle in radians."""
        transform = Rotation(
            rotation_type=RotationConfig.AXIS_ANGLE, angle_type=AngleConfig.RADIANS
        )
        axis_angle = np.array([0.0, 0.0, 0.0])
        result = transform(axis_angle)
        expected = np.array([0.0, 0.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_rotation_axis_angle_degrees(self):
        """Test Rotation with axis-angle in degrees."""
        transform = Rotation(
            rotation_type=RotationConfig.AXIS_ANGLE, angle_type=AngleConfig.DEGREES
        )
        # 90 degree rotation around z-axis
        axis_angle_deg = np.array([0.0, 0.0, 90.0])
        result = transform(axis_angle_deg)
        expected = R.from_rotvec([0.0, 0.0, np.pi / 2]).as_quat()
        np.testing.assert_array_almost_equal(result, expected)

    def test_rotation_unsupported_type(self):
        """Test Rotation with a fake rotation type to hit the error path."""
        with pytest.raises(Exception):  # Pydantic will raise validation error
            Rotation(rotation_type="unsupported")


class TestPose:
    """Tests for Pose transform."""

    def test_pose_matrix(self):
        """Test Pose with matrix format."""
        transform = Pose(pose_type=PoseConfig.MATRIX)
        # 4x4 transformation matrix (identity)
        pose_matrix = np.eye(4)
        result = transform(pose_matrix)
        # Should extract position (last column) and rotation (3x3 matrix)
        expected_pos = pose_matrix[:3, 3]
        expected_rot = R.from_matrix(pose_matrix[:3, :3]).as_quat()
        expected = np.concatenate([expected_pos, expected_rot])
        np.testing.assert_array_almost_equal(result, expected)

    def test_pose_matrix_1d(self):
        """Test Pose with matrix format as 1D array."""
        transform = Pose(pose_type=PoseConfig.MATRIX)
        # 4x4 transformation matrix as 1D array (row-major)
        pose_matrix_1d = np.eye(4).flatten()
        result = transform(pose_matrix_1d)
        pose_matrix = pose_matrix_1d.reshape(4, 4)
        expected_pos = pose_matrix[:3, 3]
        expected_rot = R.from_matrix(pose_matrix[:3, :3]).as_quat()
        expected = np.concatenate([expected_pos, expected_rot])
        np.testing.assert_array_almost_equal(result, expected)

    def test_pose_position_orientation_quaternion(self):
        """Test Pose with position_orientation and quaternion."""
        transform = Pose(
            pose_type=PoseConfig.POSITION_ORIENTATION,
            rotation_type=RotationConfig.QUATERNION,
            angle_type=AngleConfig.RADIANS,
            seq=QuaternionOrderConfig.XYZW,
        )
        # Position + quaternion (xyzw)
        pose = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])
        result = transform(pose)
        # Should just concatenate position and quaternion (already xyzw)
        expected = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_pose_position_orientation_euler(self):
        """Test Pose with position_orientation and euler."""
        transform = Pose(
            pose_type=PoseConfig.POSITION_ORIENTATION,
            rotation_type=RotationConfig.EULER,
            angle_type=AngleConfig.RADIANS,
            seq=EulerOrderConfig.XYZ,
        )
        # Position + euler angles
        pose = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
        result = transform(pose)
        expected_pos = np.array([1.0, 2.0, 3.0])
        expected_rot = R.from_euler("xyz", [0.0, 0.0, 0.0]).as_quat()
        expected = np.concatenate([expected_pos, expected_rot])
        np.testing.assert_array_almost_equal(result, expected)

    def test_pose_unsupported_type(self):
        """Test Pose with unsupported pose type."""

        class FakePoseType:
            UNSUPPORTED = "unsupported"

        with pytest.raises(ValueError, match="Input should be"):
            Pose(pose_type=FakePoseType.UNSUPPORTED)


class TestImageFormat:
    """Tests for ImageFormat transform."""

    def test_image_format_hwc(self):
        """Test ImageFormat with HWC format (no change)."""
        transform = ImageFormat(format=ImageConventionConfig.CHANNELS_LAST)
        image = np.random.rand(32, 64, 3)  # H, W, C
        result = transform(image)
        # Test shape remains unchanged
        assert result.shape == image.shape

    def test_image_format_chw(self):
        """Test ImageFormat with CHW format (convert to HWC)."""
        transform = ImageFormat(format=ImageConventionConfig.CHANNELS_FIRST)
        image = np.random.rand(3, 32, 64)  # C, H, W
        result = transform(image)
        # Should now have shape H, W, C
        assert result.shape == (32, 64, 3)


class TestImageChannelOrder:
    """Tests for ImageChannelOrder transform."""

    def test_image_channel_order_rgb(self):
        """Test ImageChannelOrder with RGB (no change)."""
        transform = ImageChannelOrder(order=ImageChannelOrderConfig.RGB)
        image = np.random.rand(32, 64, 3)
        result = transform(image)
        np.testing.assert_array_equal(result, image)

    def test_image_channel_order_bgr(self):
        """Test ImageChannelOrder with BGR (convert to RGB)."""
        transform = ImageChannelOrder(order=ImageChannelOrderConfig.BGR)
        image = np.array([[[1.0, 2.0, 3.0]]])  # B, G, R
        result = transform(image)
        expected = np.array([[[3.0, 2.0, 1.0]]])  # R, G, B
        np.testing.assert_array_equal(result, expected)


class TestCastToNumpyDtype:
    """Tests for CastToNumpyDtype transform."""

    def test_cast_to_uint8(self):
        """Test CastToNumpyDtype to uint8."""
        transform = CastToNumpyDtype(dtype=np.uint8)
        data = np.array([1.5, 2.7, 3.9], dtype=np.float32)
        result = transform(data)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, np.array([1, 2, 3], dtype=np.uint8))

    def test_cast_to_float32(self):
        """Test CastToNumpyDtype to float32."""
        transform = CastToNumpyDtype(dtype=np.float32)
        data = np.array([1, 2, 3], dtype=np.int32)
        result = transform(data)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(
            result, np.array([1.0, 2.0, 3.0], dtype=np.float32)
        )


class TestNumpyToScalar:
    """Tests for NumpyToScalar transform."""

    def test_numpy_to_scalar(self):
        """Test NumpyToScalar conversion."""
        transform = NumpyToScalar()
        data = np.array([42.5])
        result = transform(data)
        assert isinstance(result, (float, np.floating))
        assert result == 42.5

    def test_numpy_to_scalar_int(self):
        """Test NumpyToScalar with integer array."""
        transform = NumpyToScalar()
        data = np.array([42])
        result = transform(data)
        assert isinstance(result, (int, np.integer))
        assert result == 42


class TestScale:
    """Tests for Scale transform."""

    def test_scale_positive(self):
        """Test Scale with positive factor."""
        transform = Scale(factor=2.0)
        data = np.array([1.0, 2.0, 3.0])
        result = transform(data)
        expected = np.array([2.0, 4.0, 6.0])
        np.testing.assert_array_equal(result, expected)

    def test_scale_negative(self):
        """Test Scale with negative factor."""
        transform = Scale(factor=-1.0)
        data = np.array([1.0, 2.0, 3.0])
        result = transform(data)
        expected = np.array([-1.0, -2.0, -3.0])
        np.testing.assert_array_equal(result, expected)

    def test_scale_fractional(self):
        """Test Scale with fractional factor."""
        transform = Scale(factor=0.5)
        data = np.array([2.0, 4.0, 6.0])
        result = transform(data)
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result, expected)


class TestClip:
    """Tests for Clip transform."""

    def test_clip_within_range(self):
        """Test Clip with values within range."""
        transform = Clip(min=0.0, max=10.0)
        data = np.array([1.0, 5.0, 9.0])
        result = transform(data)
        np.testing.assert_array_equal(result, data)

    def test_clip_below_min(self):
        """Test Clip with values below minimum."""
        transform = Clip(min=0.0, max=10.0)
        data = np.array([-5.0, 1.0, 5.0])
        result = transform(data)
        expected = np.array([0.0, 1.0, 5.0])
        np.testing.assert_array_equal(result, expected)

    def test_clip_above_max(self):
        """Test Clip with values above maximum."""
        transform = Clip(min=0.0, max=10.0)
        data = np.array([1.0, 5.0, 15.0])
        result = transform(data)
        expected = np.array([1.0, 5.0, 10.0])
        np.testing.assert_array_equal(result, expected)

    def test_clip_both_bounds(self):
        """Test Clip with values outside both bounds."""
        transform = Clip(min=0.0, max=10.0)
        data = np.array([-5.0, 5.0, 15.0])
        result = transform(data)
        expected = np.array([0.0, 5.0, 10.0])
        np.testing.assert_array_equal(result, expected)


class TestNormalize:
    """Tests for Normalize transform."""

    def test_normalize_basic(self):
        """Test Normalize basic functionality."""
        transform = Normalize(min=0.0, max=10.0)
        data = np.array([0.0, 5.0, 10.0])
        result = transform(data)
        expected = np.array([0.0, 0.5, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_normalize_negative_range(self):
        """Test Normalize with negative range."""
        transform = Normalize(min=-1.0, max=1.0)
        data = np.array([-1.0, 0.0, 1.0])
        result = transform(data)
        expected = np.array([0.0, 0.5, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_normalize_outside_range(self):
        """Test Normalize with values outside range."""
        transform = Normalize(min=0.0, max=10.0)
        data = np.array([-5.0, 5.0, 15.0])
        result = transform(data)
        expected = np.array([-0.5, 0.5, 1.5])
        np.testing.assert_array_almost_equal(result, expected)


class TestUnnormalize:
    """Tests for Unnormalize transform."""

    def test_unnormalize_basic(self):
        """Test Unnormalize basic functionality."""
        transform = Unnormalize(min=0.0, max=10.0)
        data = np.array([0.0, 0.5, 1.0])
        result = transform(data)
        expected = np.array([0.0, 5.0, 10.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_unnormalize_negative_range(self):
        """Test Unnormalize with negative range."""
        transform = Unnormalize(min=-1.0, max=1.0)
        data = np.array([0.0, 0.5, 1.0])
        result = transform(data)
        expected = np.array([-1.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(result, expected)


class TestFlipSign:
    """Tests for FlipSign transform."""

    def test_flip_sign_positive(self):
        """Test FlipSign with positive value."""
        transform = FlipSign()
        data = 5.0
        result = transform(data)
        assert result == -5.0

    def test_flip_sign_negative(self):
        """Test FlipSign with negative value."""
        transform = FlipSign()
        data = -5.0
        result = transform(data)
        assert result == 5.0

    def test_flip_sign_zero(self):
        """Test FlipSign with zero."""
        transform = FlipSign()
        data = 0.0
        result = transform(data)
        assert result == 0.0


class TestOffset:
    """Tests for Offset transform."""

    def test_offset_positive(self):
        """Test Offset with positive value."""
        transform = Offset(value=5.0)
        data = 10.0
        result = transform(data)
        assert result == 15.0

    def test_offset_negative(self):
        """Test Offset with negative value."""
        transform = Offset(value=-5.0)
        data = 10.0
        result = transform(data)
        assert result == 5.0

    def test_offset_zero(self):
        """Test Offset with zero."""
        transform = Offset(value=0.0)
        data = 10.0
        result = transform(data)
        assert result == 10.0


class TestNanToNum:
    """Tests for NanToNum transform."""

    def test_nan_to_num_nan(self):
        """Test NanToNum with NaN values."""
        transform = NanToNum()
        data = np.array([1.0, np.nan, 3.0])
        result = transform(data)
        expected = np.array([1.0, 0.0, 3.0])
        np.testing.assert_array_equal(result, expected)

    def test_nan_to_num_inf(self):
        """Test NanToNum with infinity values."""
        transform = NanToNum()
        data = np.array([1.0, np.inf, -np.inf, 3.0])
        result = transform(data)
        expected = np.array([1.0, 0.0, 0.0, 3.0])
        np.testing.assert_array_equal(result, expected)

    def test_nan_to_num_normal(self):
        """Test NanToNum with normal values."""
        transform = NanToNum()
        data = np.array([1.0, 2.0, 3.0])
        result = transform(data)
        np.testing.assert_array_equal(result, data)


class TestDegreesToRadians:
    """Tests for DegreesToRadians transform."""

    def test_degrees_to_radians(self):
        """Test DegreesToRadians conversion."""
        transform = DegreesToRadians()
        data = np.array([0.0, 90.0, 180.0, 360.0])
        result = transform(data)
        expected = np.array([0.0, np.pi / 2, np.pi, 2 * np.pi])
        np.testing.assert_array_almost_equal(result, expected)

    def test_degrees_to_radians_negative(self):
        """Test DegreesToRadians with negative values."""
        transform = DegreesToRadians()
        data = np.array([-90.0, -180.0])
        result = transform(data)
        expected = np.array([-np.pi / 2, -np.pi])
        np.testing.assert_array_almost_equal(result, expected)


class TestLanguageFromBytes:
    """Tests for LanguageFromBytes transform."""

    def test_language_from_bytes(self):
        """Test LanguageFromBytes conversion."""
        transform = LanguageFromBytes()
        data = b"Hello, World!"
        result = transform(data)
        assert isinstance(result, str)
        assert result == "Hello, World!"

    def test_language_from_bytes_unicode(self):
        """Test LanguageFromBytes with unicode characters."""
        transform = LanguageFromBytes()
        data = "Hello, 世界!".encode()
        result = transform(data)
        assert isinstance(result, str)
        assert result == "Hello, 世界!"

    def test_language_from_bytes_empty(self):
        """Test LanguageFromBytes with empty bytes."""
        transform = LanguageFromBytes()
        data = b""
        result = transform(data)
        assert isinstance(result, str)
        assert result == ""
