"""Tests for the import session models and recording provenance."""

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from neuracore_types import (
    IMPORT_SESSION_RESERVED_FLOOR,
    IMPORT_SESSION_STRIDE,
    ImportSession,
    ImportSessionCloseRequest,
    ImportSessionCreateRequest,
    ImportSessionStatus,
    PendingRecording,
    Recording,
    RecordingSource,
)

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.mark.parametrize("worker_count", [0, IMPORT_SESSION_STRIDE + 1])
def test_create_request_rejects_worker_counts_outside_the_stride(worker_count):
    with pytest.raises(ValidationError, match="worker_count"):
        ImportSessionCreateRequest(
            robot_id="robot", dataset_id="dataset", worker_count=worker_count
        )


@pytest.mark.parametrize("worker_count", [1, IMPORT_SESSION_STRIDE])
def test_create_request_accepts_worker_counts_within_the_stride(worker_count):
    request = ImportSessionCreateRequest(
        robot_id="robot", dataset_id="dataset", worker_count=worker_count
    )
    assert request.worker_count == worker_count


@pytest.mark.parametrize(
    "status", [ImportSessionStatus.COMPLETED, ImportSessionStatus.CANCELLED]
)
def test_close_request_accepts_final_statuses(status):
    assert ImportSessionCloseRequest(status=status.value).status == status


@pytest.mark.parametrize("status", ["active", "expired"])
def test_close_request_rejects_other_statuses(status):
    with pytest.raises(ValidationError):
        ImportSessionCloseRequest(status=status)


def test_import_session_round_trips_through_json():
    session = ImportSession(
        id="session",
        org_id="org",
        robot_id="robot",
        dataset_id="dataset",
        created_by="user",
        range_index=3,
        instance_start=IMPORT_SESSION_RESERVED_FLOOR + 3 * IMPORT_SESSION_STRIDE,
        worker_count=4,
        status=ImportSessionStatus.ACTIVE,
        created_at=NOW,
        last_heartbeat_at=NOW,
        heartbeat_timeout_s=60.0,
    )
    payload = json.loads(session.model_dump_json())
    assert payload["status"] == "active"
    assert payload["closed_at"] is None
    assert ImportSession.model_validate(payload) == session


@pytest.mark.parametrize("model", [Recording, PendingRecording])
def test_recordings_without_provenance_still_validate(model):
    payload = {"id": "rec", "org_id": "org", "progress": 0}
    recording = model.model_validate(payload)
    assert recording.source is None
    assert recording.import_session_id is None


def test_pending_recording_carries_provenance_into_a_recording():
    pending = PendingRecording(
        id="rec",
        org_id="org",
        progress=100,
        source=RecordingSource.IMPORT_SESSION,
        import_session_id="session",
    )
    recording = Recording.model_validate(
        pending.model_dump(include=set(Recording.model_fields))
    )
    assert recording.source == RecordingSource.IMPORT_SESSION
    assert recording.import_session_id == "session"
