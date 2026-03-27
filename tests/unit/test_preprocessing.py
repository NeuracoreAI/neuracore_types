import pytest
from pydantic import ValidationError

from neuracore_types import DataType
from neuracore_types.preprocessing import (
    PreProcessingConfiguration,
    PreProcessingMethod,
)


def test_preprocessing_method_custom_callable_requires_custom_name():
    with pytest.raises(ValidationError, match="custom_callable is only allowed"):
        PreProcessingMethod(name="resize_pad", custom_callable="test.path.to.function")


def test_preprocessing_configuration_slot_keys_string_are_converted_to_int():
    config = PreProcessingConfiguration(
        steps={
            DataType.RGB_IMAGES: {
                "0": [PreProcessingMethod(name="resize_pad", args={"size": [64, 64]})]
            }
        }
    )

    assert 0 in config.steps[DataType.RGB_IMAGES]
    assert "0" not in config.steps[DataType.RGB_IMAGES]


def test_preprocessing_configuration_invalid_slot_key_raises():
    with pytest.raises(ValidationError):
        PreProcessingConfiguration(
            steps={
                DataType.RGB_IMAGES: {
                    "not_an_int": [
                        PreProcessingMethod(name="resize_pad", args={"size": [64, 64]})
                    ]
                }
            }
        )


def test_preprocessing_configuration_invalid_steps_structure_raises():
    with pytest.raises(ValidationError):
        PreProcessingConfiguration(steps="not_a_dict")  # type: ignore[arg-type]


def test_preprocessing_configuration_unknown_data_type_key_raises():
    with pytest.raises(ValidationError):
        PreProcessingConfiguration(
            steps={
                "unknown_data_type": {  # type: ignore[dict-item]
                    0: [PreProcessingMethod(name="resize_pad", args={"size": [64, 64]})]
                }
            }
        )
