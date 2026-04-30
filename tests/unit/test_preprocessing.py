import pytest
from pydantic import ValidationError

from neuracore_types import DataType
from neuracore_types.preprocessing import PreProcessingConfiguration


class _DummyMethod:
    @staticmethod
    def allowed_data_types() -> frozenset[DataType]:
        return frozenset({DataType.RGB_IMAGES})

    def __call__(self, data):
        return data


def test_preprocessing_configuration_slot_keys_string_are_converted_to_int():
    config = PreProcessingConfiguration(
        steps={DataType.RGB_IMAGES: {"0": [_DummyMethod()]}}
    )

    assert 0 in config.steps[DataType.RGB_IMAGES]
    assert "0" not in config.steps[DataType.RGB_IMAGES]


def test_preprocessing_configuration_invalid_slot_key_raises():
    with pytest.raises(ValidationError):
        PreProcessingConfiguration(
            steps={DataType.RGB_IMAGES: {"not_an_int": [_DummyMethod()]}}
        )


def test_preprocessing_configuration_invalid_steps_structure_raises():
    with pytest.raises(ValidationError):
        PreProcessingConfiguration(steps="not_a_dict")  # type: ignore[arg-type]


def test_preprocessing_configuration_unknown_data_type_key_raises():
    with pytest.raises(ValidationError):
        PreProcessingConfiguration(
            steps={
                "unknown_data_type": {0: [_DummyMethod()]}  # type: ignore[dict-item]
            }
        )
