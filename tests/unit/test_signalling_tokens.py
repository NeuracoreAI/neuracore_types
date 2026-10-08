"""Tests for signalling token models."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from neuracore_types import SignallingTokenRequest, SignallingTokenResponse


def test_signalling_token_request_defaults_to_organization_scope():
    """A request without a robot covers the organization."""
    assert SignallingTokenRequest().model_dump(mode="json") == {"robot_id": None}


def test_signalling_token_request_serializes_robot_id():
    """A request with a robot keeps the robot ID."""
    payload = SignallingTokenRequest(robot_id="robot123").model_dump(mode="json")

    assert payload == {"robot_id": "robot123"}


def test_signalling_token_request_rejects_empty_robot_id():
    """An empty robot ID is not a robot."""
    with pytest.raises(ValidationError):
        SignallingTokenRequest(robot_id="")


def test_signalling_token_response_round_trips_json():
    """The response keeps the token and both times through JSON."""
    response = SignallingTokenResponse(
        token="header.payload.signature",
        expires_at=datetime(2026, 10, 8, 12, 15, tzinfo=timezone.utc),
        renew_after=datetime(2026, 10, 8, 12, 10, tzinfo=timezone.utc),
    )

    restored = SignallingTokenResponse.model_validate_json(response.model_dump_json())

    assert restored == response
