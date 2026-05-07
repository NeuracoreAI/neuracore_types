"""Tests for the SUBTASK_LANGUAGE data type registration."""

from neuracore_types import DataType
from neuracore_types.nc_data import (
    DATA_TYPE_CONTENT_MAPPING,
    DATA_TYPE_TO_NC_DATA_CLASS,
    DATA_TYPE_TO_NC_DATA_IMPORT_CONFIG_CLASS,
)
from neuracore_types.nc_data.language_data import LanguageData, LanguageDataImportConfig


def test_subtask_language_datatype_exists():
    assert DataType.SUBTASK_LANGUAGE.value == "SUBTASK_LANGUAGE"


def test_subtask_language_maps_to_language_data_class():
    assert DATA_TYPE_TO_NC_DATA_CLASS[DataType.SUBTASK_LANGUAGE] is LanguageData


def test_subtask_language_maps_to_language_import_config():
    assert (
        DATA_TYPE_TO_NC_DATA_IMPORT_CONFIG_CLASS[DataType.SUBTASK_LANGUAGE]
        is LanguageDataImportConfig
    )


def test_subtask_language_content_mapping_is_json():
    assert DATA_TYPE_CONTENT_MAPPING[DataType.SUBTASK_LANGUAGE] == "JSON"
