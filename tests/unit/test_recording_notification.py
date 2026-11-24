"""Tests for recording notification models."""

import pytest
from pydantic import ValidationError

from neuracore_types import (
    BaseRecodingUpdatePayload,
    RecordingNotification,
    RecordingNotificationType,
    RecordingStartPayload,
    RecordingStopPayload,
)


def test_stop_notification_keeps_its_end_time():
    """A STOP payload with an end time parses as RecordingStopPayload."""
    notification = RecordingNotification.model_validate({
        "type": RecordingNotificationType.STOP,
        "payload": {
            "recording_id": "rec-1",
            "robot_id": "robot-1",
            "instance": 0,
            "end_time": 12.5,
        },
    })

    assert isinstance(notification.payload, RecordingStopPayload)
    assert notification.payload.end_time == 12.5


def test_payload_without_end_time_stays_a_base_payload():
    """SAVED, DISCARDED and EXPIRED payloads carry no end time."""
    notification = RecordingNotification.model_validate({
        "type": RecordingNotificationType.SAVED,
        "payload": {"recording_id": "rec-1", "robot_id": "robot-1", "instance": 0},
    })

    assert type(notification.payload) is BaseRecodingUpdatePayload


def test_an_import_disposition_parses_without_an_instance():
    """Imported recordings have no robot instance to report."""
    for notification_type in (
        RecordingNotificationType.SAVED,
        RecordingNotificationType.DISCARDED,
        RecordingNotificationType.EXPIRED,
    ):
        notification = RecordingNotification.model_validate({
            "type": notification_type,
            "payload": {"recording_id": "rec-1", "robot_id": "robot-1"},
        })

        assert type(notification.payload) is BaseRecodingUpdatePayload
        assert notification.payload.instance is None


def test_start_and_stop_payloads_require_an_instance():
    """Only live robot recordings start and stop, so both name the instance."""
    with pytest.raises(ValidationError):
        RecordingStopPayload.model_validate({
            "recording_id": "rec-1",
            "robot_id": "robot-1",
            "end_time": 12.5,
        })
    with pytest.raises(ValidationError):
        RecordingStartPayload.model_validate({
            "recording_id": "rec-1",
            "robot_id": "robot-1",
            "created_by": "user-1",
            "start_time": 1.0,
        })
