"""Tests for recording notification models."""

from neuracore_types import (
    BaseRecodingUpdatePayload,
    RecordingNotification,
    RecordingNotificationType,
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
