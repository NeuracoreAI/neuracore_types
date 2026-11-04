"""Tests for DoneData and BatchedDoneData."""

from typing import cast

import numpy as np
import pytest
import torch

from neuracore_types import BatchedDoneData, DoneData
from neuracore_types.batched_nc_data import DATA_TYPE_TO_BATCHED_NC_DATA_CLASS
from neuracore_types.importer.data_config import DataFormat, MappingItem
from neuracore_types.nc_data import (
    DATA_TYPE_CONTENT_MAPPING,
    DATA_TYPE_TO_NC_DATA_CLASS,
    DataType,
)
from neuracore_types.nc_data.done_data import DoneDataImportConfig


class TestDoneData:
    """Tests for DoneData functionality."""

    def test_sample(self):
        """Test DoneData.sample() creates a not-done instance."""
        data = DoneData.sample()
        assert isinstance(data, DoneData)
        assert data.done is False
        assert data.type == "DoneData"

    @pytest.mark.parametrize("value, expected", [(0, False), (1, True)])
    def test_accepts_zero_and_one(self, value, expected):
        """Test that 0 and 1 are coerced to booleans."""
        assert DoneData(done=value).done is expected

    def test_rejects_non_binary_value(self):
        """Test that values other than 0/1 are rejected."""
        with pytest.raises(Exception):
            DoneData(done=0.5)

    def test_serialization(self):
        """Test JSON serialization and deserialization."""
        data = DoneData(done=True)
        loaded = DoneData.model_validate_json(data.model_dump_json())

        assert loaded.done is True
        assert loaded.timestamp == data.timestamp

    @pytest.mark.parametrize("done, expected", [(False, 0.0), (True, 1.0)])
    def test_calculate_statistics(self, done, expected):
        """Test statistics report the flag as a 0/1 value."""
        stats = DoneData(done=done).calculate_statistics()

        assert stats.type == "DoneDataStats"
        assert stats.done.mean[0] == expected
        assert stats.done.min[0] == expected
        assert stats.done.max[0] == expected
        assert stats.done.count[0] == 1


class TestBatchedDoneData:
    """Tests for BatchedDoneData functionality."""

    def test_from_nc_data(self):
        """Test BatchedDoneData.from_nc_data()."""
        batched = BatchedDoneData.from_nc_data(DoneData(done=True))
        batched = cast(BatchedDoneData, batched)

        assert batched.done.shape == (1, 1, 1)
        assert batched.done.dtype == torch.float32
        assert batched.done[0, 0, 0] == 1.0

    def test_from_nc_data_list(self):
        """Test BatchedDoneData.from_nc_data_list() stacks over time."""
        batched = BatchedDoneData.from_nc_data_list(
            [DoneData(done=False), DoneData(done=False), DoneData(done=True)]
        )

        assert batched.done.shape == (1, 3, 1)
        assert batched.done[0, :, 0].tolist() == [0.0, 0.0, 1.0]

    def test_sample(self):
        """Test BatchedDoneData.sample()."""
        batched = BatchedDoneData.sample(batch_size=3, time_steps=2)
        assert batched.done.shape == (3, 2, 1)
        assert torch.all(batched.done == 0.0)

    def test_can_serialize_deserialize(self):
        """Test JSON serialization and deserialization."""
        batched = BatchedDoneData.from_nc_data_list(
            [DoneData(done=False), DoneData(done=True)]
        )
        loaded = BatchedDoneData.model_validate_json(batched.model_dump_json())

        assert torch.equal(loaded.done, batched.done)


class TestDoneDataRegistration:
    """Tests that DataType.DONE is wired into the registries."""

    def test_mappings(self):
        """Test DONE maps to the done classes and JSON content."""
        assert DATA_TYPE_TO_NC_DATA_CLASS[DataType.DONE] is DoneData
        assert DATA_TYPE_TO_BATCHED_NC_DATA_CLASS[DataType.DONE] is BatchedDoneData
        assert DATA_TYPE_CONTENT_MAPPING[DataType.DONE] == "JSON"

    def test_import_config_converts_to_scalar(self):
        """Test the import transforms turn a numpy flag into a scalar."""
        config = DoneDataImportConfig(
            source="done",
            format=DataFormat(),
            mapping=[MappingItem(name="done")],
        )
        transforms = config.mapping[0].transforms

        assert transforms(np.array([1])) == 1
