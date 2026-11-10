"""Tests for the regional signalling fields of the shared models.

Each region lists its own robots with their registration epochs.
Clients merge the lists of every region endpoint. All new fields are optional,
so legacy payloads still validate.
"""

import pytest
from pydantic import ValidationError

from neuracore_types import (
    AvailableRobotCapacityInit,
    AvailableRobotCapacityUpdate,
    AvailableRobotInstance,
    SignallingTokenRequest,
    SignallingTokenResponse,
)

LEGACY_INIT = {
    "type": "init",
    "robots": [{
        "robot_id": "robot-1",
        "instances": {"0": {"robot_instance": 0, "tracks": {}, "connections": 1}},
    }],
}


def _optional(model: type, field: str) -> bool:
    schema = model.model_json_schema()
    return field in schema["properties"] and field not in schema.get("required", [])


def test_available_robot_instance_accepts_epoch():
    """A robot instance entry keeps its registration epoch through JSON."""
    entry = AvailableRobotInstance(robot_instance=0, tracks={}, connections=0, epoch=3)

    restored = AvailableRobotInstance.model_validate_json(entry.model_dump_json())

    assert restored.epoch == 3


def test_legacy_availability_payload_still_validates():
    """A payload with no epochs validates, with no epoch and no stream epochs."""
    init = AvailableRobotCapacityInit.model_validate(LEGACY_INIT)

    assert init.robots[0].instances[0].epoch is None
    assert init.robots[0].instances[0].stream_epochs == {}


def test_capacity_update_carries_instance_epochs():
    """An update maps each robot instance in it to its epoch."""
    update = AvailableRobotCapacityUpdate.model_validate(
        {"type": "update", "epochs": {"robot-1": {"0": 5, "1": 2}}}
    )

    assert update.epochs == {"robot-1": {0: 5, 1: 2}}
    assert AvailableRobotCapacityUpdate().epochs == {}


def test_available_robot_instance_carries_stream_epochs():
    """A robot instance entry maps each of its streams to its epoch."""
    entry = AvailableRobotInstance(
        robot_instance=0,
        tracks={},
        connections=0,
        stream_epochs={"stream-a": 4, "stream-b": 7},
    )

    restored = AvailableRobotInstance.model_validate_json(entry.model_dump_json())

    assert restored.stream_epochs == {"stream-a": 4, "stream-b": 7}


def test_capacity_update_carries_stream_epochs():
    """An update maps each stream in it to its epoch."""
    update = AvailableRobotCapacityUpdate.model_validate(
        {"type": "update", "stream_epochs": {"stream-a": 4, "stream-b": 7}}
    )

    assert update.stream_epochs == {"stream-a": 4, "stream-b": 7}
    assert AvailableRobotCapacityUpdate().stream_epochs == {}


def test_token_response_accepts_region_endpoints():
    """The token response lists region endpoints; absent means none."""
    base = {
        "token": "header.payload.signature",
        "expires_at": "2026-10-30T12:15:00Z",
        "renew_after": "2026-10-30T12:10:00Z",
    }
    endpoints = ["https://us.signalling.example", "https://eu.signalling.example"]

    assert SignallingTokenResponse.model_validate(base).region_endpoints == []
    listed = SignallingTokenResponse.model_validate(
        {**base, "region_endpoints": endpoints}
    )
    assert listed.region_endpoints == endpoints


def test_token_request_accepts_robot_instance_and_register_epoch():
    """A robot asks for a register token for one robot instance."""
    request = SignallingTokenRequest(
        robot_id="robot-1", robot_instance=1, register_epoch=True
    )

    assert request.model_dump(mode="json") == {
        "robot_id": "robot-1",
        "robot_instance": 1,
        "register_epoch": True,
    }
    assert SignallingTokenRequest(robot_id="robot-1").register_epoch is False


def test_token_request_rejects_a_negative_robot_instance():
    """A robot instance is a non-negative integer."""
    with pytest.raises(ValidationError):
        SignallingTokenRequest(robot_id="robot-1", robot_instance=-1)


def test_new_fields_are_optional_in_the_json_schema():
    """The TypeScript types, generated from the JSON schema, mark them optional."""
    assert _optional(AvailableRobotInstance, "epoch")
    assert _optional(AvailableRobotInstance, "stream_epochs")
    assert _optional(AvailableRobotCapacityUpdate, "stream_epochs")
    assert _optional(AvailableRobotCapacityUpdate, "epochs")
    assert _optional(SignallingTokenResponse, "region_endpoints")
    assert _optional(SignallingTokenRequest, "robot_instance")
    assert _optional(SignallingTokenRequest, "register_epoch")


def test_epochs_reject_negative_values():
    """An epoch is a non-negative integer, per instance and per stream."""
    with pytest.raises(ValidationError):
        AvailableRobotInstance(robot_instance=0, tracks={}, connections=0, epoch=-1)
    with pytest.raises(ValidationError):
        AvailableRobotCapacityUpdate(type="update", epochs={"robot-1": {0: -1}})
    with pytest.raises(ValidationError):
        AvailableRobotInstance(
            robot_instance=0, tracks={}, connections=0, stream_epochs={"s": -1}
        )
    with pytest.raises(ValidationError):
        AvailableRobotCapacityUpdate(type="update", stream_epochs={"s": -1})


def test_microsecond_epoch_survives_a_json_round_trip():
    """A microsecond epoch keeps its exact value through JSON."""
    epoch = 1_800_000_000_123_456
    update = AvailableRobotCapacityUpdate(type="update", epochs={"robot-1": {0: epoch}})

    parsed = AvailableRobotCapacityUpdate.model_validate_json(update.model_dump_json())

    assert parsed.epochs == {"robot-1": {0: epoch}}
