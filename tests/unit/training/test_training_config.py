"""Unit tests for training, preprocessing and algorithm config models."""

import sys

import pytest
from pydantic import ValidationError

from neuracore_types import (
    AlgorithmConfig,
    ColorJitterConfig,
    DataType,
    PreprocessingStage,
    ResizePadConfig,
    SynchronizationConfig,
    TrainingConfig,
)


def test_training_config_defaults_match_hydra_file():
    training_config = TrainingConfig()
    assert training_config.epochs == 100
    assert training_config.batch_size == "auto"
    rgb_input_methods = training_config.preprocessing["input"][DataType.RGB_IMAGES]
    assert rgb_input_methods[1] == ColorJitterConfig(
        stages=[PreprocessingStage.TRAINING]
    )


def test_training_config_rejects_unknown_key():
    with pytest.raises(ValidationError):
        TrainingConfig.model_validate({"hidden_dim": 512})


@pytest.mark.parametrize("invalid_batch_size", [0, "large"])
def test_training_config_rejects_invalid_batch_size(invalid_batch_size):
    with pytest.raises(ValidationError):
        TrainingConfig.model_validate({"batch_size": invalid_batch_size})


def test_preprocessing_rejects_unknown_method_name():
    with pytest.raises(ValidationError):
        TrainingConfig.model_validate(
            {"preprocessing": {"input": {"RGB_IMAGES": [{"method": "os.system"}]}}}
        )


def test_preprocessing_round_trips_through_json():
    training_config = TrainingConfig.model_validate({
        "preprocessing": {
            "input": {
                "RGB_IMAGES": [
                    {"method": "resize_pad", "size": [128, 96], "stages": ["inference"]}
                ]
            }
        }
    })
    reloaded = TrainingConfig.model_validate(training_config.model_dump(mode="json"))
    assert reloaded.preprocessing["input"][DataType.RGB_IMAGES] == [
        ResizePadConfig(size=(128, 96), stages=[PreprocessingStage.INFERENCE])
    ]


@pytest.mark.parametrize("invalid_stages", [[], ["validation"]])
def test_preprocessing_rejects_invalid_stages(invalid_stages):
    with pytest.raises(ValidationError):
        ResizePadConfig.model_validate({"stages": invalid_stages})


def test_algorithm_config_keeps_model_arguments():
    algorithm_config = AlgorithmConfig.model_validate({
        "hidden_dim": 512,
        "clip_grad_norm": 1.0,
    })
    assert algorithm_config.model_dump() == {"hidden_dim": 512, "clip_grad_norm": 1.0}


@pytest.mark.parametrize("misplaced_key", ["seed", "epochs", "frequency", "_target_"])
def test_algorithm_config_rejects_misplaced_keys(misplaced_key):
    with pytest.raises(ValidationError, match=misplaced_key):
        AlgorithmConfig.model_validate({misplaced_key: 1})


def test_training_config_schema_marks_every_field_required():
    json_schema = TrainingConfig.model_json_schema()
    assert set(json_schema["required"]) == set(TrainingConfig.model_fields)


def test_synchronization_frequency_defaults_to_ten():
    assert SynchronizationConfig().frequency == 10


@pytest.mark.parametrize("non_integer_value", [True, "5", 2.5])
def test_integer_settings_reject_non_integers(non_integer_value):
    with pytest.raises(ValidationError):
        TrainingConfig.model_validate(
            {"validation_rollout_frequency": non_integer_value}
        )
    with pytest.raises(ValidationError):
        TrainingConfig.model_validate({"batch_size": non_integer_value})


def test_synchronization_config_defaults_to_the_largest_max_delay():
    assert SynchronizationConfig().max_delay_s == sys.float_info.max
