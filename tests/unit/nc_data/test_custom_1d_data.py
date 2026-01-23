"""Tests for Custom1DData and BatchedCustom1DData."""

import json

import numpy as np
import torch

from neuracore_types import BatchedCustom1DData, Custom1DData
from neuracore_types.importer.data_config import MappingItem
from neuracore_types.importer.transform import FlipSign, Offset
from neuracore_types.nc_data.custom_1d_data import Custom1DDataImportConfig


class TestCustom1DData:
    """Tests for Custom1DData functionality."""

    def test_sample(self):
        """Test Custom1DData.sample() creates valid instance."""
        data = Custom1DData.sample()
        assert isinstance(data, Custom1DData)
        assert isinstance(data.data, np.ndarray)
        assert data.data.shape == (10,)
        assert data.data.dtype == np.float32
        assert data.type == "Custom1DData"

    def test_serialization(self):
        """Test JSON serialization of custom 1D data."""
        arr = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
        data = Custom1DData(data=arr)

        json_str = data.model_dump_json()
        loaded = Custom1DData.model_validate_json(json_str)

        assert np.allclose(loaded.data, data.data)

    def test_list_encoding_decoding(self):
        """Test list encoding/decoding."""
        arr = np.random.randn(20).astype(np.float32)
        data = Custom1DData(data=arr)

        # Serialize to dict
        data_dict = json.loads(data.model_dump_json())
        assert isinstance(data_dict["data"], list)

        # Deserialize
        loaded = Custom1DData.model_validate(data_dict)
        assert np.allclose(loaded.data, arr)

    def test_single_element_array(self):
        """Test custom 1D data with single element."""
        arr = np.array([5.0], dtype=np.float32)
        data = Custom1DData(data=arr)

        assert data.data.shape == (1,)
        assert data.data[0] == 5.0

    def test_large_array(self):
        """Test custom 1D data with large array."""
        arr = np.random.randn(10000).astype(np.float32)
        data = Custom1DData(data=arr)

        assert data.data.shape == (10000,)

    def test_empty_array(self):
        """Test handling of empty array."""
        arr = np.array([], dtype=np.float32)
        data = Custom1DData(data=arr)

        assert data.data.shape == (0,)

    def test_all_zeros(self):
        """Test custom 1D data with all zeros."""
        arr = np.zeros(50, dtype=np.float32)
        data = Custom1DData(data=arr)

        assert np.allclose(data.data, 0.0)

    def test_all_ones(self):
        """Test custom 1D data with all ones."""
        arr = np.ones(50, dtype=np.float32)
        data = Custom1DData(data=arr)

        assert np.allclose(data.data, 1.0)


class TestBatchedCustom1DData:
    """Tests for BatchedCustom1DData functionality."""

    def test_from_nc_data(self):
        """Test BatchedCustom1DData.from_nc_data() conversion."""
        custom_data = Custom1DData(data=np.array([1.0, 2.0, 3.0], dtype=np.float32))
        batched = BatchedCustom1DData.from_nc_data(custom_data)

        assert isinstance(batched, BatchedCustom1DData)
        assert batched.data.shape == (1, 1, 3)
        assert torch.allclose(batched.data[0, 0], torch.tensor([1.0, 2.0, 3.0]))

    def test_sample(self):
        """Test BatchedCustom1DData.sample() with different dimensions."""
        batched = BatchedCustom1DData.sample(batch_size=2, time_steps=3)
        assert batched.data.shape == (2, 3, 10)

    def test_sample_single_dimension(self):
        """Test sample with single batch and timestep."""
        batched = BatchedCustom1DData.sample(batch_size=1, time_steps=1)
        assert batched.data.shape == (1, 1, 10)

    def test_sample_large_dimensions(self):
        """Test sample with large dimensions."""
        batched = BatchedCustom1DData.sample(batch_size=50, time_steps=100)
        assert batched.data.shape == (50, 100, 10)

    def test_to_device(self):
        """Test moving to different device."""
        batched = BatchedCustom1DData.sample(batch_size=2, time_steps=3)
        batched_cpu = batched.to(torch.device("cpu"))

        assert batched_cpu.data.device.type == "cpu"
        assert torch.allclose(batched_cpu.data, batched.data)

    def test_from_nc_data_preserves_values(self):
        """Test that from_nc_data preserves exact values."""
        arr = np.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype=np.float32)
        custom_data = Custom1DData(data=arr)
        batched = BatchedCustom1DData.from_nc_data(custom_data)

        assert torch.allclose(batched.data[0, 0], torch.from_numpy(arr), rtol=1e-5)

    def test_can_serialize_deserialize(self):
        """Test JSON serialization and deserialization of BatchedCustom1DData."""
        batched = BatchedCustom1DData.sample(batch_size=2, time_steps=2)
        json_str = batched.model_dump_json()
        loaded = BatchedCustom1DData.model_validate_json(json_str)

        assert torch.equal(loaded.data, batched.data)
        assert loaded.data.shape == batched.data.shape


class TestCustom1DDataSerialization:
    """Tests for serialization of Custom1DData."""

    def test_roundtrip_serialization(self):
        """Test that data survives roundtrip serialization."""
        original_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        data = Custom1DData(data=original_arr)

        # Serialize
        json_str = data.model_dump_json()

        # Deserialize
        loaded = Custom1DData.model_validate_json(json_str)

        # Verify
        assert np.allclose(loaded.data, original_arr)

    def test_serialization_preserves_dtype(self):
        """Test that serialization preserves float32 dtype."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        data = Custom1DData(data=arr)

        json_str = data.model_dump_json()
        loaded = Custom1DData.model_validate_json(json_str)

        assert loaded.data.dtype == np.float32

    def test_base64_is_reversible(self):
        """Test that base64 encoding is fully reversible."""
        arr = np.random.randn(100).astype(np.float32)
        data = Custom1DData(data=arr)

        # Encode
        encoded = data.model_dump()

        # Decode
        loaded = Custom1DData.model_validate(encoded)

        # Should be exactly equal (not just close)
        assert np.array_equal(loaded.data, arr)


class TestCustom1DDataImportConfig:
    """Tests for Custom1DDataImportConfig class."""

    def test_custom_1d_data_import_config_basic(self):
        """Test Custom1DDataImportConfig basic functionality."""
        data_point = Custom1DDataImportConfig(
            source="custom",
            mapping=[MappingItem(name="custom_value")],
        )
        transforms = data_point.mapping[0].transforms.transforms
        assert len(transforms) == 0

    def test_custom_1d_data_import_config_inverted(self):
        """Test Custom1DDataImportConfig with inverted flag."""
        data_point = Custom1DDataImportConfig(
            source="custom",
            mapping=[MappingItem(name="custom_value", inverted=True)],
        )
        transforms = data_point.mapping[0].transforms.transforms
        assert any(isinstance(t, FlipSign) for t in transforms)

    def test_custom_1d_data_import_config_offset(self):
        """Test Custom1DDataImportConfig with offset."""
        data_point = Custom1DDataImportConfig(
            source="custom",
            mapping=[MappingItem(name="custom_value", offset=2.5)],
        )
        transforms = data_point.mapping[0].transforms.transforms
        assert any(isinstance(t, Offset) for t in transforms)
