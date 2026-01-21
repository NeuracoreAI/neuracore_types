"""Tests for ParallelGripperOpenAmountData and BatchedParallelGripperOpenAmountData."""

from typing import cast

import numpy as np
import pytest
import torch

from neuracore_types import (
    BatchedParallelGripperOpenAmountData,
    ParallelGripperOpenAmountData,
)
from neuracore_types.batched_nc_data import DATA_TYPE_TO_BATCHED_NC_DATA_CLASS
from neuracore_types.importer.config import NormalizeConfig
from neuracore_types.importer.data_config import DataFormat, MappingItem
from neuracore_types.importer.transform import Clip, Normalize
from neuracore_types.nc_data import DATA_TYPE_TO_NC_DATA_CLASS, DataType
from neuracore_types.nc_data.parallel_gripper_open_amount_data import (
    ParallelGripperOpenAmountDataImportConfig,
)


class TestParallelGripperOpenAmountData:
    """Tests for ParallelGripperOpenAmountData functionality."""

    def test_sample(self):
        """Test ParallelGripperOpenAmountData.sample() creates valid instance."""
        data = ParallelGripperOpenAmountData.sample()
        assert isinstance(data, ParallelGripperOpenAmountData)
        assert isinstance(data.open_amount, float)
        assert data.type == "ParallelGripperOpenAmountData"

    def test_calculate_statistics(self):
        """Test calculate_statistics() returns valid stats."""
        data = ParallelGripperOpenAmountData(open_amount=0.5)
        stats = data.calculate_statistics()

        assert stats.type == "ParallelGripperOpenAmountDataStats"
        assert stats.open_amount is not None
        assert len(stats.open_amount.mean) == 1
        assert stats.open_amount.mean[0] == 0.5

    def test_value_range_validation(self):
        """Test that extreme values are handled correctly."""
        data = ParallelGripperOpenAmountData(open_amount=1.0)
        assert data.open_amount == 1.0

        data = ParallelGripperOpenAmountData(open_amount=0.0)
        assert data.open_amount == 0.0

    def test_negative_gripper_value(self):
        """Test handling of negative gripper values."""
        data = ParallelGripperOpenAmountData(open_amount=-0.5)
        assert data.open_amount == -0.5

    def test_serialization(self):
        """Test JSON serialization and deserialization."""
        data = ParallelGripperOpenAmountData(open_amount=0.75)
        json_str = data.model_dump_json()
        loaded = ParallelGripperOpenAmountData.model_validate_json(json_str)

        assert loaded.open_amount == data.open_amount
        assert loaded.timestamp == data.timestamp

    def test_invalid_type(self):
        """Test that invalid value type raises error."""
        with pytest.raises(Exception):
            ParallelGripperOpenAmountData(open_amount="not a number")


class TestBatchedParallelGripperOpenAmountData:
    """Tests for BatchedParallelGripperOpenAmountData functionality."""

    def test_from_nc_data(self):
        """Test BatchedParallelGripperOpenAmountData.from_nc_data()."""
        gripper_data = ParallelGripperOpenAmountData(open_amount=0.75)
        batched = BatchedParallelGripperOpenAmountData.from_nc_data(gripper_data)

        assert isinstance(batched, BatchedParallelGripperOpenAmountData)
        assert batched.open_amount.shape == (1, 1, 1)
        assert batched.open_amount[0, 0, 0] == 0.75

    def test_sample(self):
        """Test BatchedParallelGripperOpenAmountData.sample()."""
        batched = BatchedParallelGripperOpenAmountData.sample(
            batch_size=3, time_steps=2
        )
        assert batched.open_amount.shape == (3, 2, 1)

    def test_sample_large_dimensions(self):
        """Test sample with large dimensions."""
        batched = BatchedParallelGripperOpenAmountData.sample(
            batch_size=10, time_steps=20
        )
        assert batched.open_amount.shape == (10, 20, 1)

    def test_to_device(self):
        """Test moving to different device."""
        batched = BatchedParallelGripperOpenAmountData.sample(
            batch_size=2, time_steps=3
        )
        batched_cpu = batched.to(torch.device("cpu"))
        batched_cpu = cast(BatchedParallelGripperOpenAmountData, batched_cpu)

        assert batched_cpu.open_amount.device.type == "cpu"
        assert torch.allclose(batched_cpu.open_amount, batched.open_amount)

    def test_from_nc_data_preserves_value(self):
        """Test that from_nc_data preserves exact value."""
        test_value = 0.12345
        gripper_data = ParallelGripperOpenAmountData(open_amount=test_value)
        batched = BatchedParallelGripperOpenAmountData.from_nc_data(gripper_data)
        batched = cast(BatchedParallelGripperOpenAmountData, batched)

        assert torch.isclose(
            batched.open_amount[0, 0, 0], torch.tensor(test_value), rtol=1e-5
        )

    def test_can_serialize_deserialize(self):
        """Test JSON serialization and deserialization."""
        batched = BatchedParallelGripperOpenAmountData.sample(
            batch_size=2, time_steps=2
        )
        json_str = batched.model_dump_json()
        loaded = BatchedParallelGripperOpenAmountData.model_validate_json(json_str)

        assert torch.equal(loaded.open_amount, batched.open_amount)
        assert loaded.open_amount.shape == batched.open_amount.shape


class TestParallelGripperStatistics:
    """Tests for ParallelGripperOpenAmountData statistics."""

    def test_statistics_values(self):
        """Test that statistics contain correct values."""
        data = ParallelGripperOpenAmountData(open_amount=0.8)
        stats = data.calculate_statistics()

        assert np.isclose(stats.open_amount.mean[0], 0.8)
        assert stats.open_amount.count[0] == 1
        assert np.isclose(stats.open_amount.min[0], 0.8)
        assert np.isclose(stats.open_amount.max[0], 0.8)

    def test_statistics_concatenation(self):
        """Test that gripper statistics can be concatenated."""
        data1 = ParallelGripperOpenAmountData(open_amount=0.3)
        data2 = ParallelGripperOpenAmountData(open_amount=0.7)

        stats1 = data1.calculate_statistics()
        stats2 = data2.calculate_statistics()

        concatenated = stats1.open_amount.concatenate(stats2.open_amount)
        assert len(concatenated.mean) == 2
        assert np.isclose(concatenated.mean[0], 0.3)
        assert np.isclose(concatenated.mean[1], 0.7)


class TestParallelGripperTargetOpenAmountDataType:
    """Tests for PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS DataType mapping."""

    def test_target_data_type_exists(self):
        """Test that PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS DataType exists."""
        assert hasattr(DataType, "PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS")
        assert (
            DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS.value
            == "PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS"
        )

    def test_target_data_type_mapped_to_nc_data_class(self):
        """Test that target DataType is mapped to ParallelGripperOpenAmountData."""
        assert (
            DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS in DATA_TYPE_TO_NC_DATA_CLASS
        )
        assert (
            DATA_TYPE_TO_NC_DATA_CLASS[DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS]
            == ParallelGripperOpenAmountData
        )

    def test_target_data_type_mapped_to_batched_nc_data_class(self):
        """Test target DataType is mapped to BatchedParallelGripperOpenAmountData."""
        assert (
            DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS
            in DATA_TYPE_TO_BATCHED_NC_DATA_CLASS
        )
        assert (
            DATA_TYPE_TO_BATCHED_NC_DATA_CLASS[
                DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS
            ]
            == BatchedParallelGripperOpenAmountData
        )

    def test_target_uses_same_data_class_as_state(self):
        """Test that target and state use the same underlying data classes."""
        # Similar to how JOINT_TARGET_POSITIONS uses JointData
        assert (
            DATA_TYPE_TO_NC_DATA_CLASS[DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS]
            == DATA_TYPE_TO_NC_DATA_CLASS[DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS]
        )
        assert (
            DATA_TYPE_TO_BATCHED_NC_DATA_CLASS[
                DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS
            ]
            == DATA_TYPE_TO_BATCHED_NC_DATA_CLASS[
                DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS
            ]
        )


class TestParallelGripperOpenAmountDataImportConfig:
    """Tests for ParallelGripperOpenAmountDataImportConfig class."""

    def test_parallel_gripper_transforms_no_normalize(self):
        """Test ParallelGripperOpenAmountDataImportConfig transforms w/o normalize."""
        data_point = ParallelGripperOpenAmountDataImportConfig(
            source="gripper",
            mapping=[MappingItem(name="gripper_open")],
            format=DataFormat(normalize=None),
        )
        transforms = data_point.mapping[0].transforms.transforms
        assert not any(isinstance(t, Normalize) for t in transforms)
        assert any(isinstance(t, Clip) for t in transforms)

    def test_parallel_gripper_transforms_with_normalize(self):
        """Test ParallelGripperOpenAmountDataImportConfig transforms with normalize."""
        normalize = NormalizeConfig(min=0.0, max=100.0)
        data_point = ParallelGripperOpenAmountDataImportConfig(
            source="gripper",
            mapping=[MappingItem(name="gripper_open")],
            format=DataFormat(normalize=normalize),
        )
        transforms = data_point.mapping[0].transforms.transforms
        assert any(isinstance(t, Normalize) for t in transforms)
