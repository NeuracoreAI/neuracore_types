"""Training settings shared by the API, the frontend and training runs."""

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, Strict, WithJsonSchema

from neuracore_types.nc_data import DataType
from neuracore_types.utils.pydantic_to_ts import (
    REQUIRED_WITH_DEFAULT_FLAG,
    fix_required_with_defaults,
)

_STRICT_MODEL_CONFIG = ConfigDict(
    extra="forbid", json_schema_extra=fix_required_with_defaults
)


class PreprocessingStage(str, Enum):
    """Part of a run that applies a preprocessing method.

    TRAINING covers the training split. INFERENCE covers validation and the
    packaged model.
    """

    TRAINING = "training"
    INFERENCE = "inference"


def all_preprocessing_stages() -> list[PreprocessingStage]:
    """Return every preprocessing stage."""
    return [PreprocessingStage.TRAINING, PreprocessingStage.INFERENCE]


def training_only_stages() -> list[PreprocessingStage]:
    """Return the stage list of a method that only the training split applies."""
    return [PreprocessingStage.TRAINING]


PreprocessingStages = Annotated[list[PreprocessingStage], Field(min_length=1)]


class ResizePadConfig(BaseModel):
    """Settings for resizing frames to a target size with symmetric padding.

    Attributes:
        method: Name of the preprocessing method.
        stages: Stages of the run that apply the method.
        size: Target size as [height, width] in pixels.
    """

    model_config = _STRICT_MODEL_CONFIG

    method: Literal["resize_pad"] = Field(
        default="resize_pad", json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    stages: PreprocessingStages = Field(
        default_factory=all_preprocessing_stages,
        json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG,
    )
    size: tuple[PositiveInt, PositiveInt] = Field(
        default=(224, 224), json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )


class ColorJitterConfig(BaseModel):
    """Settings for random colour changes applied to RGB frames.

    Attributes:
        method: Name of the preprocessing method.
        stages: Stages of the run that apply the method.
        brightness: Maximum relative brightness change.
        contrast: Maximum relative contrast change.
        saturation: Maximum relative saturation change.
        hue: Maximum hue shift.
    """

    model_config = _STRICT_MODEL_CONFIG

    method: Literal["color_jitter"] = Field(
        default="color_jitter", json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    stages: PreprocessingStages = Field(
        default_factory=all_preprocessing_stages,
        json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG,
    )
    brightness: float = Field(
        default=0.2, ge=0, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    contrast: float = Field(
        default=0.2, ge=0, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    saturation: float = Field(
        default=0.2, ge=0, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    hue: float = Field(
        default=0.05, ge=0, le=0.5, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )


class GaussianNoiseConfig(BaseModel):
    """Settings for Gaussian noise added to RGB frames.

    Attributes:
        method: Name of the preprocessing method.
        stages: Stages of the run that apply the method.
        std: Standard deviation of the noise in pixel units on the 0 to 255 scale.
        clip: Clamp the noisy frame to the 0 to 255 range.
    """

    model_config = _STRICT_MODEL_CONFIG

    method: Literal["gaussian_noise"] = Field(
        default="gaussian_noise", json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    stages: PreprocessingStages = Field(
        default_factory=all_preprocessing_stages,
        json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG,
    )
    std: float = Field(default=5.0, ge=0, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG)
    clip: bool = Field(default=True, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG)


BatchSize = Annotated[
    Annotated[PositiveInt, Strict()] | Literal["auto"],
    WithJsonSchema(
        {
            "anyOf": [
                {"type": "integer", "exclusiveMinimum": 0},
                {"const": "auto"},
            ]
        }
    ),
]

PreprocessingMethodConfig = Annotated[
    ResizePadConfig | ColorJitterConfig | GaussianNoiseConfig,
    Field(discriminator="method"),
]

PreprocessingPipeline = dict[
    Literal["input", "output"], dict[DataType, list[PreprocessingMethodConfig]]
]


def default_preprocessing() -> PreprocessingPipeline:
    """Return the preprocessing a run applies by default."""
    return {
        "input": {
            DataType.RGB_IMAGES: [
                ResizePadConfig(),
                ColorJitterConfig(stages=training_only_stages()),
                GaussianNoiseConfig(stages=training_only_stages()),
            ],
            DataType.DEPTH_IMAGES: [ResizePadConfig()],
        },
        "output": {
            DataType.RGB_IMAGES: [ResizePadConfig()],
            DataType.DEPTH_IMAGES: [ResizePadConfig()],
        },
    }


class TrainingConfig(BaseModel):
    """Settings that control how a training run executes.

    Attributes:
        seed: Random seed for data splits and model initialisation.
        epochs: Number of epochs to train for.
        batch_size: Samples per gradient update, or auto to tune it.
        output_prediction_horizon: Length of the predicted action sequence.
        validation_split: Fraction of the dataset held out for validation.
        validation_rollout_points: Validation points saved as policy
            rollouts. 0 disables rollout saving.
        validation_rollout_frequency: Epochs between saved rollouts.
        logging_frequency: Steps between logged model metrics.
        system_logging_frequency: Steps between host and accelerator
            utilisation samples. 0 disables them.
        keep_last_n_checkpoints: Number of most recent checkpoints to keep.
        checkpoint_saving_frequency: Keep checkpoints whose epoch is divisible
            by this.
        device: Torch device to train on, or None to pick one automatically.
        num_train_workers: Data loader workers for the training split.
        num_val_workers: Data loader workers for the validation split.
        max_prefetch_decode_workers: Threads that decode videos during
            prefetch.
        num_concurrent_prefetch_requests: Network requests the prefetch keeps
            outstanding at once.
        dataset_sample_cache: Cache fully built training samples on disk.
        preprocessing: Preprocessing methods per input and output data type,
            each tagged with the stages that apply it.
    """

    model_config = _STRICT_MODEL_CONFIG

    seed: int = Field(
        strict=True, default=42, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    epochs: int = Field(
        strict=True, default=100, ge=1, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    batch_size: BatchSize = Field(
        default="auto", json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    output_prediction_horizon: int = Field(
        strict=True, default=100, ge=1, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    validation_split: float = Field(
        default=0.2, gt=0, lt=1, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    validation_rollout_points: int = Field(
        strict=True, default=16, ge=0, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    validation_rollout_frequency: int = Field(
        strict=True, default=5, ge=1, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    logging_frequency: int = Field(
        strict=True, default=50, ge=1, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    system_logging_frequency: int = Field(
        strict=True, default=200, ge=0, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    keep_last_n_checkpoints: int = Field(
        strict=True, default=2, ge=1, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    checkpoint_saving_frequency: int = Field(
        strict=True, default=5, ge=1, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    device: str | None = Field(
        default=None, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    num_train_workers: int = Field(
        strict=True, default=4, ge=0, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    num_val_workers: int = Field(
        strict=True, default=2, ge=0, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    max_prefetch_decode_workers: int = Field(
        strict=True, default=4, ge=1, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    num_concurrent_prefetch_requests: int = Field(
        strict=True, default=16, ge=1, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    dataset_sample_cache: bool = Field(
        default=True, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    preprocessing: PreprocessingPipeline = Field(
        default_factory=default_preprocessing,
        json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG,
    )
