"""Tests for pending recording cancellation types and public exports."""

import json

import pytest
from pydantic import ValidationError

from neuracore_types import PendingRecording, PendingRecordingsMetadata, Recording


def test_pending_recording_defaults_cancelled_to_false():
    recording = PendingRecording(id="recording-1", org_id="org-1", progress=0)

    assert recording.cancelled is False
    for payload in (recording.model_dump(), json.loads(recording.model_dump_json())):
        assert payload["cancelled"] is False
        assert payload["deleted"] is False
        assert payload["expected_trace_count"] == 0


def test_pending_recording_accepts_cancelled():
    recording = PendingRecording(
        id="recording-1", org_id="org-1", progress=0, cancelled=True
    )

    assert recording.cancelled is True
    assert recording.model_dump()["cancelled"] is True
    assert json.loads(recording.model_dump_json())["cancelled"] is True


def test_cancelled_schema_is_required_only_for_pending_recordings():
    schema = PendingRecording.model_json_schema()

    assert "cancelled" in schema["required"]
    assert schema["properties"]["cancelled"]["type"] == "boolean"
    assert schema["properties"]["cancelled"]["default"] is False
    assert "cancelled" not in Recording.model_fields


@pytest.mark.parametrize("recording_ids", [[], ["recording-1", "recording-2"]])
def test_pending_recordings_metadata_accepts_lists(recording_ids):
    metadata = PendingRecordingsMetadata(cancellable_recording_ids=recording_ids)

    assert metadata.cancellable_recording_ids == recording_ids
    assert metadata.model_dump() == {"cancellable_recording_ids": recording_ids}


@pytest.mark.parametrize("payload", [{}, {"cancellable_recording_ids": None}])
def test_pending_recordings_metadata_requires_list(payload):
    with pytest.raises(ValidationError):
        PendingRecordingsMetadata.model_validate(payload)
