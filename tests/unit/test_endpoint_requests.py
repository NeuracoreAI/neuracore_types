"""Tests for endpoint request models."""

from neuracore_types import DataType
from neuracore_types.endpoints.endpoint_requests import DeploymentRequest


def test_deployment_request_serializes_canonical_embodiment_description_fields():
    """DeploymentRequest should keep the current deploy API payload shape."""
    payload = DeploymentRequest(
        training_id="training123",
        name="endpoint-name",
        ttl=1800,
        input_embodiment_description={
            DataType.RGB_IMAGES: {0: "camera1"},
        },
        output_embodiment_description={
            DataType.JOINT_TARGET_POSITIONS: {0: "joint1"},
        },
    ).model_dump(mode="json")

    assert payload == {
        "training_id": "training123",
        "name": "endpoint-name",
        "ttl": 1800,
        "input_embodiment_description": {
            DataType.RGB_IMAGES.value: {"0": "camera1"},
        },
        "output_embodiment_description": {
            DataType.JOINT_TARGET_POSITIONS.value: {"0": "joint1"},
        },
        "config": {
            "machine_type": None,
            "gpu_type": "NVIDIA_TESLA_T4",
            "gpu_count": 1,
        },
    }
