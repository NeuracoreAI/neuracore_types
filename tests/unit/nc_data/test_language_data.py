"""Tests for LanguageData and BatchedLanguageData."""

import pytest
import torch

from neuracore_types import BatchedLanguageData, LanguageData
from neuracore_types.importer.config import LanguageConfig
from neuracore_types.importer.data_config import DataFormat, MappingItem
from neuracore_types.importer.transform import LanguageFromBytes
from neuracore_types.nc_data.language_data import LanguageDataImportConfig


class TestLanguageData:
    """Tests for LanguageData functionality."""

    def test_sample(self):
        """Test LanguageData.sample() creates valid instance."""
        data = LanguageData.sample()
        assert isinstance(data, LanguageData)
        assert isinstance(data.text, str)
        assert len(data.text) > 0
        assert data.type == "LanguageData"

    def test_serialization(self):
        """Test JSON serialization and deserialization."""
        data = LanguageData(text="Test message")
        json_str = data.model_dump_json()
        loaded = LanguageData.model_validate_json(json_str)

        assert loaded.text == data.text
        assert loaded.timestamp == data.timestamp

    def test_empty_text(self):
        """Test handling of empty text."""
        data = LanguageData(text="")
        assert data.text == ""

    def test_very_long_text(self):
        """Test handling of very long text."""
        long_text = "A" * 10000
        data = LanguageData(text=long_text)
        assert len(data.text) == 10000

    def test_special_characters(self):
        """Test handling of special characters in text."""
        special_text = "Hello\nWorld\t🤖\u00e9"
        data = LanguageData(text=special_text)
        assert data.text == special_text

    def test_invalid_type(self):
        """Test that invalid text type raises error."""
        with pytest.raises(Exception):
            LanguageData(text=123)


class TestBatchedLanguageData:
    """Tests for BatchedLanguageData functionality."""

    def test_sample(self):
        """Test BatchedLanguageData.sample() with different dimensions."""
        batched = BatchedLanguageData.sample(batch_size=3, time_steps=5)
        assert isinstance(batched, BatchedLanguageData)
        assert batched.input_ids.shape == (3, 5, 512)

    def test_sample_single_dimension(self):
        """Test BatchedLanguageData.sample() with single dimension."""
        batched = BatchedLanguageData.sample(batch_size=1, time_steps=1)
        assert batched.input_ids.shape == (1, 1, 512)

    def test_sample_large_dimensions(self):
        """Test BatchedLanguageData.sample() with large dimensions."""
        batched = BatchedLanguageData.sample(batch_size=10, time_steps=20)
        assert batched.input_ids.shape == (10, 20, 512)

    def test_can_serialize_deserialize(self):
        """Test JSON serialization and deserialization of BatchedLanguageData."""
        batched = BatchedLanguageData.sample(batch_size=2, time_steps=2)
        json_str = batched.model_dump_json()
        loaded = BatchedLanguageData.model_validate_json(json_str)

        assert torch.equal(loaded.input_ids, batched.input_ids)
        assert loaded.input_ids.shape == batched.input_ids.shape


class TestLanguageDataImportConfig:
    """Tests for LanguageDataImportConfig class."""

    def test_language_data_import_config_string(self):
        """Test LanguageDataImportConfig with string type."""
        data_point = LanguageDataImportConfig(
            source="language",
            mapping=[MappingItem(name="instruction")],
            format=DataFormat(language_type=LanguageConfig.STRING),
        )
        transforms = data_point.mapping[0].transforms.transforms
        assert len(transforms) == 0

    def test_language_data_import_config_bytes(self):
        """Test LanguageDataImportConfig with bytes type."""
        data_point = LanguageDataImportConfig(
            source="language",
            mapping=[MappingItem(name="instruction")],
            format=DataFormat(language_type=LanguageConfig.BYTES),
        )
        transforms = data_point.mapping[0].transforms.transforms
        assert isinstance(transforms[0], LanguageFromBytes)
