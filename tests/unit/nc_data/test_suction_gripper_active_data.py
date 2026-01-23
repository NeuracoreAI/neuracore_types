"""Tests for SuctionGripperActiveData and BatchedSuctionGripperActiveData."""

from typing import cast

import numpy as np
import pytest
import torch

from neuracore_types import BatchedSuctionGripperActiveData, SuctionGripperActiveData
from neuracore_types.batched_nc_data import DATA_TYPE_TO_BATCHED_NC_DATA_CLASS
from neuracore_types.nc_data import DATA_TYPE_TO_NC_DATA_CLASS, DataType


class TestSuctionGripperActiveData:
    """Tests for SuctionGripperActiveData functionality."""

    def test_sample(self):
        """Test SuctionGripperActiveData.sample() creates valid instance."""
        data = SuctionGripperActiveData.sample()
        assert isinstance(data, SuctionGripperActiveData)
        assert isinstance(data.active, bool)
        assert data.type == "SuctionGripperActiveData"

    def test_calculate_statistics_active_true(self):
        """Test calculate_statistics() returns valid stats when active is True."""
        data = SuctionGripperActiveData(active=True)
        stats = data.calculate_statistics()

        assert stats.type == "SuctionGripperActiveDataStats"
        assert stats.active is not None
        assert len(stats.active.mean) == 1
        assert stats.active.mean[0] == 1.0

    def test_calculate_statistics_active_false(self):
        """Test calculate_statistics() returns valid stats when active is False."""
        data = SuctionGripperActiveData(active=False)
        stats = data.calculate_statistics()

        assert stats.type == "SuctionGripperActiveDataStats"
        assert stats.active is not None
        assert len(stats.active.mean) == 1
        assert stats.active.mean[0] == 0.0

    def test_active_true(self):
        """Test that active=True is handled correctly."""
        data = SuctionGripperActiveData(active=True)
        assert data.active is True

    def test_active_false(self):
        """Test that active=False is handled correctly."""
        data = SuctionGripperActiveData(active=False)
        assert data.active is False

    def test_serialization(self):
        """Test JSON serialization and deserialization."""
        data = SuctionGripperActiveData(active=True)
        json_str = data.model_dump_json()
        loaded = SuctionGripperActiveData.model_validate_json(json_str)

        assert loaded.active == data.active
        assert loaded.timestamp == data.timestamp

    def test_serialization_false(self):
        """Test JSON serialization and deserialization with False."""
        data = SuctionGripperActiveData(active=False)
        json_str = data.model_dump_json()
        loaded = SuctionGripperActiveData.model_validate_json(json_str)

        assert loaded.active == data.active
        assert loaded.timestamp == data.timestamp

    def test_invalid_type(self):
        """Test that invalid value type raises error."""
        with pytest.raises(Exception):
            SuctionGripperActiveData(active="not a boolean")


class TestBatchedSuctionGripperActiveData:
    """Tests for BatchedSuctionGripperActiveData functionality."""

    def test_from_nc_data_true(self):
        """Test BatchedSuctionGripperActiveData.from_nc_data() with True."""
        gripper_data = SuctionGripperActiveData(active=True)
        batched = BatchedSuctionGripperActiveData.from_nc_data(gripper_data)

        assert isinstance(batched, BatchedSuctionGripperActiveData)
        assert batched.active.shape == (1, 1, 1)
        assert batched.active[0, 0, 0].item() is True

    def test_from_nc_data_false(self):
        """Test BatchedSuctionGripperActiveData.from_nc_data() with False."""
        gripper_data = SuctionGripperActiveData(active=False)
        batched = BatchedSuctionGripperActiveData.from_nc_data(gripper_data)

        assert isinstance(batched, BatchedSuctionGripperActiveData)
        assert batched.active.shape == (1, 1, 1)
        assert batched.active[0, 0, 0].item() is False

    def test_sample(self):
        """Test BatchedSuctionGripperActiveData.sample()."""
        batched = BatchedSuctionGripperActiveData.sample(batch_size=3, time_steps=2)
        assert batched.active.shape == (3, 2, 1)

    def test_sample_large_dimensions(self):
        """Test sample with large dimensions."""
        batched = BatchedSuctionGripperActiveData.sample(batch_size=10, time_steps=20)
        assert batched.active.shape == (10, 20, 1)

    def test_to_device(self):
        """Test moving to different device."""
        batched = BatchedSuctionGripperActiveData.sample(batch_size=2, time_steps=3)
        batched_cpu = batched.to(torch.device("cpu"))
        batched_cpu = cast(BatchedSuctionGripperActiveData, batched_cpu)

        assert batched_cpu.active.device.type == "cpu"
        assert torch.equal(batched_cpu.active, batched.active)

    def test_from_nc_data_preserves_value_true(self):
        """Test that from_nc_data preserves exact value (True)."""
        gripper_data = SuctionGripperActiveData(active=True)
        batched = BatchedSuctionGripperActiveData.from_nc_data(gripper_data)
        batched = cast(BatchedSuctionGripperActiveData, batched)

        assert batched.active[0, 0, 0].item() is True

    def test_from_nc_data_preserves_value_false(self):
        """Test that from_nc_data preserves exact value (False)."""
        gripper_data = SuctionGripperActiveData(active=False)
        batched = BatchedSuctionGripperActiveData.from_nc_data(gripper_data)
        batched = cast(BatchedSuctionGripperActiveData, batched)

        assert batched.active[0, 0, 0].item() is False

    def test_can_serialize_deserialize(self):
        """Test JSON serialization and deserialization."""
        batched = BatchedSuctionGripperActiveData.sample(batch_size=2, time_steps=2)
        json_str = batched.model_dump_json()
        loaded = BatchedSuctionGripperActiveData.model_validate_json(json_str)

        assert torch.equal(loaded.active, batched.active)
        assert loaded.active.shape == batched.active.shape

    def test_from_nc_data_list(self):
        """Test from_nc_data_list creates correct batched data."""
        data_list = [
            SuctionGripperActiveData(active=True),
            SuctionGripperActiveData(active=False),
            SuctionGripperActiveData(active=True),
        ]
        batched = BatchedSuctionGripperActiveData.from_nc_data_list(data_list)

        assert batched.active.shape == (1, 3, 1)
        assert batched.active[0, 0, 0].item() is True
        assert batched.active[0, 1, 0].item() is False
        assert batched.active[0, 2, 0].item() is True


class TestSuctionGripperStatistics:
    """Tests for SuctionGripperActiveData statistics."""

    def test_statistics_values_true(self):
        """Test that statistics contain correct values for True."""
        data = SuctionGripperActiveData(active=True)
        stats = data.calculate_statistics()

        assert np.isclose(stats.active.mean[0], 1.0)
        assert stats.active.count[0] == 1
        assert np.isclose(stats.active.min[0], 1.0)
        assert np.isclose(stats.active.max[0], 1.0)

    def test_statistics_values_false(self):
        """Test that statistics contain correct values for False."""
        data = SuctionGripperActiveData(active=False)
        stats = data.calculate_statistics()

        assert np.isclose(stats.active.mean[0], 0.0)
        assert stats.active.count[0] == 1
        assert np.isclose(stats.active.min[0], 0.0)
        assert np.isclose(stats.active.max[0], 0.0)

    def test_statistics_concatenation(self):
        """Test that suction gripper statistics can be concatenated."""
        data1 = SuctionGripperActiveData(active=False)
        data2 = SuctionGripperActiveData(active=True)

        stats1 = data1.calculate_statistics()
        stats2 = data2.calculate_statistics()

        concatenated = stats1.active.concatenate(stats2.active)
        assert len(concatenated.mean) == 2
        assert np.isclose(concatenated.mean[0], 0.0)
        assert np.isclose(concatenated.mean[1], 1.0)


class TestSuctionGripperTargetActiveDataType:
    """Tests for SUCTION_GRIPPER_TARGET_ACTIVES DataType mapping."""

    def test_target_data_type_exists(self):
        """Test that SUCTION_GRIPPER_TARGET_ACTIVES DataType exists."""
        assert hasattr(DataType, "SUCTION_GRIPPER_TARGET_ACTIVES")
        assert (
            DataType.SUCTION_GRIPPER_TARGET_ACTIVES.value
            == "SUCTION_GRIPPER_TARGET_ACTIVES"
        )

    def test_target_data_type_mapped_to_nc_data_class(self):
        """Test that target DataType is mapped to SuctionGripperActiveData."""
        assert DataType.SUCTION_GRIPPER_TARGET_ACTIVES in DATA_TYPE_TO_NC_DATA_CLASS
        assert (
            DATA_TYPE_TO_NC_DATA_CLASS[DataType.SUCTION_GRIPPER_TARGET_ACTIVES]
            == SuctionGripperActiveData
        )

    def test_target_data_type_mapped_to_batched_nc_data_class(self):
        """Test target DataType is mapped to BatchedSuctionGripperActiveData."""
        assert (
            DataType.SUCTION_GRIPPER_TARGET_ACTIVES
            in DATA_TYPE_TO_BATCHED_NC_DATA_CLASS
        )
        assert (
            DATA_TYPE_TO_BATCHED_NC_DATA_CLASS[DataType.SUCTION_GRIPPER_TARGET_ACTIVES]
            == BatchedSuctionGripperActiveData
        )

    def test_target_uses_same_data_class_as_state(self):
        """Test that target and state use the same underlying data classes."""
        # Similar to how JOINT_TARGET_POSITIONS uses JointData
        assert (
            DATA_TYPE_TO_NC_DATA_CLASS[DataType.SUCTION_GRIPPER_TARGET_ACTIVES]
            == DATA_TYPE_TO_NC_DATA_CLASS[DataType.SUCTION_GRIPPER_ACTIVES]
        )
        assert (
            DATA_TYPE_TO_BATCHED_NC_DATA_CLASS[DataType.SUCTION_GRIPPER_TARGET_ACTIVES]
            == DATA_TYPE_TO_BATCHED_NC_DATA_CLASS[DataType.SUCTION_GRIPPER_ACTIVES]
        )


class TestSuctionGripperActiveDataType:
    """Tests for SUCTION_GRIPPER_ACTIVES DataType mapping."""

    def test_data_type_exists(self):
        """Test that SUCTION_GRIPPER_ACTIVES DataType exists."""
        assert hasattr(DataType, "SUCTION_GRIPPER_ACTIVES")
        assert DataType.SUCTION_GRIPPER_ACTIVES.value == "SUCTION_GRIPPER_ACTIVES"

    def test_data_type_mapped_to_nc_data_class(self):
        """Test that DataType is mapped to SuctionGripperActiveData."""
        assert DataType.SUCTION_GRIPPER_ACTIVES in DATA_TYPE_TO_NC_DATA_CLASS
        assert (
            DATA_TYPE_TO_NC_DATA_CLASS[DataType.SUCTION_GRIPPER_ACTIVES]
            == SuctionGripperActiveData
        )

    def test_data_type_mapped_to_batched_nc_data_class(self):
        """Test DataType is mapped to BatchedSuctionGripperActiveData."""
        assert DataType.SUCTION_GRIPPER_ACTIVES in DATA_TYPE_TO_BATCHED_NC_DATA_CLASS
        assert (
            DATA_TYPE_TO_BATCHED_NC_DATA_CLASS[DataType.SUCTION_GRIPPER_ACTIVES]
            == BatchedSuctionGripperActiveData
        )
