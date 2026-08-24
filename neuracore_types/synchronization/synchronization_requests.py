"""Synchronization request models."""

from enum import Enum

from pydantic import BaseModel

from neuracore_types.synchronization.synchronization import SynchronizationDetails


class SynchronizeDatasetRequest(BaseModel):
    """Request model for synchronizing a dataset.

    Attributes:
        dataset_id: Identifier of the dataset to synchronize.
        synchronization_details: Details for how to perform the synchronization.
    """

    dataset_id: str
    synchronization_details: SynchronizationDetails


class SynchronizedRecordingStatus(str, Enum):
    """Lifecycle stage of an asynchronous recording synchronization."""

    PENDING = "PENDING"
    READY = "READY"
    FAILED = "FAILED"


class SynchronizeRecordingRequest(BaseModel):
    """Request model for synchronizing a recording.

    Attributes:
        recording_id: The ID of the recording to synchronize.
        synchronization_details: Details for how to perform the synchronization.
    """

    recording_id: str
    synchronization_details: SynchronizationDetails
