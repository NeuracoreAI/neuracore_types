"""Synchronization request models."""

from enum import Enum

from pydantic import BaseModel, Field

from neuracore_types.synchronization.synchronization import (
    SYNCHRONIZATION_CONFIG_INPUT_NAMES,
    SynchronizationConfig,
)


class SynchronizeDatasetRequest(BaseModel):
    """Request model for synchronizing a dataset.

    Attributes:
        dataset_id: Identifier of the dataset to synchronize.
        synchronization_config: Details for how to perform the synchronization.
    """

    dataset_id: str
    synchronization_config: SynchronizationConfig = Field(
        validation_alias=SYNCHRONIZATION_CONFIG_INPUT_NAMES
    )


class SynchronizeRecordingStatus(str, Enum):
    """Lifecycle stage of an asynchronous recording synchronization."""

    PENDING = "PENDING"
    READY = "READY"
    FAILED = "FAILED"


class SynchronizeRecordingRequest(BaseModel):
    """Request model for synchronizing a recording.

    Attributes:
        recording_id: The ID of the recording to synchronize.
        synchronization_config: Details for how to perform the synchronization.
    """

    recording_id: str
    synchronization_config: SynchronizationConfig = Field(
        validation_alias=SYNCHRONIZATION_CONFIG_INPUT_NAMES
    )


class SynchronizeRecordingStartResponse(BaseModel):
    """Tracking information returned when recording synchronization starts."""

    recording_id: str
    synchronized_recording_id: str
    status: SynchronizeRecordingStatus = SynchronizeRecordingStatus.PENDING


class SynchronizeRecordingProgress(BaseModel):
    """Current preparation state returned by the polling endpoint."""

    recording_id: str
    synchronize_recording_id: str
    status: SynchronizeRecordingStatus
    download_url: str | None = None
    error: str | None = None
