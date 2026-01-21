"""Tests for PoseData and BatchedPoseData."""

import numpy as np
import pytest
import torch

from neuracore_types import BatchedPoseData, PoseData
from neuracore_types.importer.config import (
    AngleConfig,
    EulerOrderConfig,
    IndexRangeConfig,
    OrientationConfig,
    PoseConfig,
    QuaternionOrderConfig,
    RotationConfig,
)
from neuracore_types.importer.data_config import DataFormat, MappingItem
from neuracore_types.importer.transform import Pose
from neuracore_types.nc_data.pose_data import PoseDataImportConfig


class TestPoseData:
    """Tests for PoseData functionality."""

    def test_sample(self):
        """Test PoseData.sample() creates valid instance."""
        data = PoseData.sample()
        assert isinstance(data, PoseData)
        assert len(data.pose) == 7
        assert data.type == "PoseData"

    def test_calculate_statistics(self):
        """Test calculate_statistics() for pose data."""
        pose = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])
        data = PoseData(pose=pose)
        stats = data.calculate_statistics()

        assert stats.type == "PoseDataStats"
        assert stats.pose is not None
        assert np.allclose(stats.pose.mean, pose)
        assert len(stats.pose.count) == 7

    def test_serialization(self):
        """Test JSON serialization and deserialization."""
        pose = np.array([1.0, 2.0, 3.0, 0.5, 0.5, 0.5, 0.5])
        data = PoseData(pose=pose)

        json_str = data.model_dump_json()
        loaded = PoseData.model_validate_json(json_str)

        assert np.allclose(loaded.pose, data.pose)

    def test_invalid_pose_length(self):
        """Test that invalid pose length raises error."""
        with pytest.raises(Exception):
            # Pose should be length 7
            PoseData(pose=np.array([1.0, 2.0, 3.0]))

    def test_pose_with_identity_quaternion(self):
        """Test pose with identity quaternion."""
        data = PoseData(pose=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
        assert data.pose[-1] == 1.0

    def test_pose_with_normalized_quaternion(self):
        """Test pose with normalized quaternion."""
        # Quaternion: [0.5, 0.5, 0.5, 0.5] (normalized)
        data = PoseData(pose=np.array([1.0, 2.0, 3.0, 0.5, 0.5, 0.5, 0.5]))
        quat = data.pose[3:]
        quat_norm = np.sqrt(sum(q**2 for q in quat))
        assert np.isclose(quat_norm, 1.0)

    def test_pose_position_components(self):
        """Test that position components are correctly stored."""
        pose = np.array([1.5, 2.5, 3.5, 0.0, 0.0, 0.0, 1.0])
        data = PoseData(pose=pose)

        assert data.pose[0] == 1.5  # x
        assert data.pose[1] == 2.5  # y
        assert data.pose[2] == 3.5  # z


class TestBatchedPoseData:
    """Tests for BatchedPoseData functionality."""

    def test_from_nc_data(self):
        """Test BatchedPoseData.from_nc_data() conversion."""
        pose_data = PoseData(
            pose=np.array([1.0, 2.0, 3.0, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        )
        batched = BatchedPoseData.from_nc_data(pose_data)

        assert isinstance(batched, BatchedPoseData)
        assert batched.pose.shape == (1, 1, 7)
        assert torch.allclose(batched.pose[0, 0], torch.tensor(pose_data.pose))

    def test_sample(self):
        """Test BatchedPoseData.sample() with different dimensions."""
        batched = BatchedPoseData.sample(batch_size=5, time_steps=2)
        assert batched.pose.shape == (5, 2, 7)

    def test_sample_single_dimension(self):
        """Test sample with single batch and timestep."""
        batched = BatchedPoseData.sample(batch_size=1, time_steps=1)
        assert batched.pose.shape == (1, 1, 7)

    def test_sample_large_dimensions(self):
        """Test sample with large dimensions."""
        batched = BatchedPoseData.sample(batch_size=100, time_steps=50)
        assert batched.pose.shape == (100, 50, 7)

    def test_to_device(self):
        """Test moving BatchedPoseData to different device."""
        batched = BatchedPoseData.sample(batch_size=2, time_steps=3)
        batched_cpu = batched.to(torch.device("cpu"))

        assert batched_cpu.pose.device.type == "cpu"
        assert torch.allclose(batched_cpu.pose, batched.pose)

    def test_from_nc_data_preserves_values(self):
        """Test that from_nc_data preserves exact values."""
        pose = np.array([1.5, 2.5, 3.5, 0.1, 0.2, 0.3, 0.9], dtype=np.float32)
        pose_data = PoseData(pose=pose)
        batched = BatchedPoseData.from_nc_data(pose_data)

        for i, val in enumerate(pose):
            assert torch.isclose(batched.pose[0, 0, i], torch.tensor(val), rtol=1e-5)

    def test_can_serialize_deserialize(self):
        """Test JSON serialization and deserialization of BatchedPoseData."""
        batched = BatchedPoseData.sample(batch_size=2, time_steps=2)
        json_str = batched.model_dump_json()
        loaded = BatchedPoseData.model_validate_json(json_str)

        assert torch.equal(loaded.pose, batched.pose)
        assert loaded.pose.shape == batched.pose.shape


class TestPoseDataStatistics:
    """Tests for PoseData statistics."""

    def test_statistics_mean(self):
        """Test that statistics mean matches pose values."""
        pose = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])
        data = PoseData(pose=pose)
        stats = data.calculate_statistics()

        assert np.allclose(stats.pose.mean, pose)

    def test_statistics_count(self):
        """Test that statistics count is 1 for all elements."""
        data = PoseData.sample()
        stats = data.calculate_statistics()

        assert all(count == 1 for count in stats.pose.count)

    def test_statistics_min_max(self):
        """Test that min and max equal mean for single sample."""
        pose = np.array([5.0, 6.0, 7.0, 0.1, 0.2, 0.3, 0.9])
        data = PoseData(pose=pose)
        stats = data.calculate_statistics()

        assert np.allclose(stats.pose.min, pose)
        assert np.allclose(stats.pose.max, pose)

    def test_statistics_std(self):
        """Test that std is 1.0 for all elements in single sample."""
        pose = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])
        data = PoseData(pose=pose)
        stats = data.calculate_statistics()

        assert np.allclose(stats.pose.std, 0.0)

    def test_statistics_concatenation(self):
        """Test that pose statistics can be concatenated."""
        data1 = PoseData(pose=np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]))
        data2 = PoseData(pose=np.array([2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 1.0]))

        stats1 = data1.calculate_statistics()
        stats2 = data2.calculate_statistics()

        concatenated = stats1.pose.concatenate(stats2.pose)
        assert len(concatenated.mean) == 14  # 7 + 7


class TestPoseDataImportConfig:
    """Tests for PoseDataImportConfig class."""

    def test_pose_data_import_config_matrix(self):
        """Test PoseDataImportConfig with matrix format."""
        index_range = IndexRangeConfig(start=0, end=16)
        data_point = PoseDataImportConfig(
            source="pose",
            mapping=[MappingItem(name="end_effector_pose", index_range=index_range)],
            format=DataFormat(pose_type=PoseConfig.MATRIX),
        )
        transforms = data_point.mapping[0].transforms.transforms
        assert any(isinstance(t, Pose) for t in transforms)

    def test_pose_data_import_config_position_orientation_quaternion(self):
        """Test PoseDataImportConfig with position_orientation and quaternion."""
        index_range = IndexRangeConfig(start=0, end=7)
        orientation = OrientationConfig(
            type=RotationConfig.QUATERNION,
            quaternion_order=QuaternionOrderConfig.WXYZ,
            angle_units=AngleConfig.RADIANS,
        )
        data_point = PoseDataImportConfig(
            source="pose",
            mapping=[MappingItem(name="end_effector_pose", index_range=index_range)],
            format=DataFormat(
                pose_type=PoseConfig.POSITION_ORIENTATION, orientation=orientation
            ),
        )
        transforms = data_point.mapping[0].transforms.transforms
        assert any(isinstance(t, Pose) for t in transforms)

    def test_pose_data_import_config_position_orientation_euler(self):
        """Test PoseDataImportConfig with position_orientation and euler."""
        index_range = IndexRangeConfig(start=0, end=6)
        orientation = OrientationConfig(
            type=RotationConfig.EULER,
            euler_order=EulerOrderConfig.ZYX,
            angle_units=AngleConfig.DEGREES,
        )
        data_point = PoseDataImportConfig(
            source="pose",
            mapping=[MappingItem(name="end_effector_pose", index_range=index_range)],
            format=DataFormat(
                pose_type=PoseConfig.POSITION_ORIENTATION, orientation=orientation
            ),
        )
        transforms = data_point.mapping[0].transforms.transforms
        assert any(isinstance(t, Pose) for t in transforms)

    def test_pose_data_import_config_requires_orientation(self):
        """Test validation that 'orientation' is required for position_orientation."""
        index_range = IndexRangeConfig(start=0, end=7)
        with pytest.raises(
            ValueError,
            match="orientation must be provided when format is 'position_orientation'",
        ):
            PoseDataImportConfig(
                source="pose",
                mapping=[
                    MappingItem(name="end_effector_pose", index_range=index_range)
                ],
                format=DataFormat(
                    pose_type=PoseConfig.POSITION_ORIENTATION, orientation=None
                ),
            )

    def test_pose_data_import_config_requires_index_range(self):
        """Test PoseDataImportConfig validation requires index_range."""
        orientation = OrientationConfig(type=RotationConfig.QUATERNION)
        with pytest.raises(
            ValueError, match="index_range is required for pose data points"
        ):
            PoseDataImportConfig(
                source="pose",
                mapping=[MappingItem(name="end_effector_pose")],
                format=DataFormat(
                    pose_type=PoseConfig.POSITION_ORIENTATION, orientation=orientation
                ),
            )

    def test_pose_data_import_config_quaternion_index_range_length(self):
        """Test PoseDataImportConfig validation for quaternion index_range length."""
        index_range = IndexRangeConfig(start=0, end=6)  # Should be 7
        orientation = OrientationConfig(type=RotationConfig.QUATERNION)
        with pytest.raises(
            ValueError,
            match="Index range length must be 7 for orientation type QUATERNION",
        ):
            PoseDataImportConfig(
                source="pose",
                mapping=[
                    MappingItem(name="end_effector_pose", index_range=index_range)
                ],
                format=DataFormat(
                    pose_type=PoseConfig.POSITION_ORIENTATION, orientation=orientation
                ),
            )

    def test_pose_data_import_config_euler_index_range_length(self):
        """Test PoseDataImportConfig validation for euler index_range length."""
        index_range = IndexRangeConfig(start=0, end=5)  # Should be 6
        orientation = OrientationConfig(type=RotationConfig.EULER)
        with pytest.raises(
            ValueError, match="Index range length must be 6 for orientation type EULER"
        ):
            PoseDataImportConfig(
                source="pose",
                mapping=[
                    MappingItem(name="end_effector_pose", index_range=index_range)
                ],
                format=DataFormat(
                    pose_type=PoseConfig.POSITION_ORIENTATION, orientation=orientation
                ),
            )
