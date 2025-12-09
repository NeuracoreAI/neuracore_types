"""Tests for EndEffectorPoseData and BatchedEndEffectorPoseData."""

import numpy as np
import pytest
import torch

from neuracore_types import BatchedEndEffectorPoseData, EndEffectorPoseData


class TestEndEffectorPoseData:
    """Tests for EndEffectorPoseData functionality."""

    def test_sample(self):
        """Test EndEffectorPoseData.sample() creates valid instance."""
        data = EndEffectorPoseData.sample()
        assert isinstance(data, EndEffectorPoseData)
        assert len(data.pose) == 7  # x, y, z, qx, qy, qz, qw
        assert data.type == "EndEffectorPoseData"

    def test_calculate_statistics(self):
        """Test calculate_statistics() returns valid stats."""
        data = EndEffectorPoseData(pose=np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]))
        stats = data.calculate_statistics()

        assert stats.type == "EndEffectorPoseDataStats"
        assert stats.pose is not None
        assert len(stats.pose.mean) == 7
        assert len(stats.pose.std) == 7
        assert len(stats.pose.count) == 7
        assert np.allclose(stats.pose.mean, data.pose)

    def test_serialization(self):
        """Test JSON serialization preserves pose data."""
        data = EndEffectorPoseData(pose=np.array([1.0, 2.0, 3.0, 0.5, 0.5, 0.5, 0.5]))
        json_str = data.model_dump_json()
        loaded = EndEffectorPoseData.model_validate_json(json_str)

        assert np.allclose(loaded.pose, data.pose)

    def test_invalid_pose_length(self):
        """Test that invalid pose length raises error."""
        with pytest.raises(Exception):
            # Pose should be length 7
            EndEffectorPoseData(pose=np.array([1.0, 2.0, 3.0]))

    def test_pose_with_identity_quaternion(self):
        """Test pose with identity quaternion."""
        data = EndEffectorPoseData(pose=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
        assert data.pose[-1] == 1.0

    def test_pose_with_normalized_quaternion(self):
        """Test pose with normalized quaternion."""
        # Quaternion: [0.5, 0.5, 0.5, 0.5] (normalized)
        data = EndEffectorPoseData(pose=np.array([1.0, 2.0, 3.0, 0.5, 0.5, 0.5, 0.5]))
        quat = data.pose[3:]
        quat_norm = np.sqrt(sum(q**2 for q in quat))
        assert np.isclose(quat_norm, 1.0)


class TestBatchedEndEffectorPoseData:
    """Tests for BatchedEndEffectorPoseData functionality."""

    def test_from_nc_data(self):
        """Test BatchedEndEffectorPoseData.from_nc_data() conversion."""
        pose_data = EndEffectorPoseData(
            pose=np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        )
        batched = BatchedEndEffectorPoseData.from_nc_data(pose_data)

        assert isinstance(batched, BatchedEndEffectorPoseData)
        assert batched.pose.shape == (1, 1, 7)
        assert torch.allclose(batched.pose[0, 0], torch.tensor(pose_data.pose))

    def test_sample(self):
        """Test BatchedEndEffectorPoseData.sample() with different dimensions."""
        batched = BatchedEndEffectorPoseData.sample(batch_size=2, time_steps=4)
        assert batched.pose.shape == (2, 4, 7)

    def test_sample_single_timestep(self):
        """Test sample with single timestep."""
        batched = BatchedEndEffectorPoseData.sample(batch_size=5, time_steps=1)
        assert batched.pose.shape == (5, 1, 7)

    def test_to_device(self):
        """Test moving BatchedEndEffectorPoseData to different device."""
        batched = BatchedEndEffectorPoseData.sample(batch_size=2, time_steps=3)
        batched_cpu = batched.to(torch.device("cpu"))

        assert batched_cpu.pose.device.type == "cpu"
        assert torch.allclose(batched_cpu.pose, batched.pose)

    def test_from_nc_data_preserves_values(self):
        """Test that from_nc_data preserves exact values."""
        pose = [1.5, 2.5, 3.5, 0.1, 0.2, 0.3, 0.9]
        pose_data = EndEffectorPoseData(pose=pose)
        batched = BatchedEndEffectorPoseData.from_nc_data(pose_data)

        for i, val in enumerate(pose):
            assert torch.isclose(batched.pose[0, 0, i], torch.tensor(val), rtol=1e-5)

    def test_can_serialize_deserialize(self):
        """Test JSON serialization and deserialization of BatchedEndEffectorPoseData."""
        batched = BatchedEndEffectorPoseData.sample(batch_size=2, time_steps=2)
        json_str = batched.model_dump_json()
        loaded = BatchedEndEffectorPoseData.model_validate_json(json_str)

        assert torch.equal(loaded.pose, batched.pose)
        assert loaded.pose.shape == batched.pose.shape


class TestEndEffectorPoseDataStatistics:
    """Tests for EndEffectorPoseData statistics."""

    def test_statistics_mean(self):
        """Test that statistics mean matches pose values."""
        pose = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])
        data = EndEffectorPoseData(pose=pose)
        stats = data.calculate_statistics()

        assert np.allclose(stats.pose.mean, pose)

    def test_statistics_count(self):
        """Test that statistics count is 1 for all elements."""
        data = EndEffectorPoseData.sample()
        stats = data.calculate_statistics()

        assert all(count == 1 for count in stats.pose.count)

    def test_statistics_min_max(self):
        """Test that min and max equal mean for single sample."""
        pose = np.array([5.0, 6.0, 7.0, 0.1, 0.2, 0.3, 0.9])
        data = EndEffectorPoseData(pose=pose)
        stats = data.calculate_statistics()

        assert np.allclose(stats.pose.min, pose)
        assert np.allclose(stats.pose.max, pose)

    def test_statistics_concatenation(self):
        """Test that pose statistics can be concatenated."""
        data1 = EndEffectorPoseData(pose=np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]))
        data2 = EndEffectorPoseData(pose=np.array([2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 1.0]))

        stats1 = data1.calculate_statistics()
        stats2 = data2.calculate_statistics()

        concatenated = stats1.pose.concatenate(stats2.pose)
        assert len(concatenated.mean) == 14  # 7 + 7
