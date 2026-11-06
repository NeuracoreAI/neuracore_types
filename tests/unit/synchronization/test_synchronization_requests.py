"""Unit tests for synchronization request models."""

from neuracore_types import SynchronizeDatasetRequest


def test_synchronize_dataset_request_accepts_synchronization_details_name():
    request = SynchronizeDatasetRequest.model_validate({
        "dataset_id": "dataset-id",
        "synchronization_details": {"frequency": 25},
    })
    assert request.synchronization_config.frequency == 25
    assert "synchronization_config" in request.model_dump()
