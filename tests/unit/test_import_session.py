"""Tests for the import session models and recording provenance."""

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from neuracore_types import (
    IMPORT_SESSION_ITEM_KEY_MAX_LENGTH,
    ImportSession,
    ImportSessionCreateRequest,
    ImportSessionItemKeysPage,
    ImportSessionItemState,
    ImportSessionSource,
    PendingRecording,
    Recording,
    RecordingStartRequest,
    RobotInstanceSource,
)

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
START = {"robot_id": "robot", "dataset_id": "dataset", "start_time": 1.0}


@pytest.mark.parametrize("expected_recording_count", [0, -1])
def test_create_request_needs_a_positive_expected_count(expected_recording_count):
    with pytest.raises(ValidationError, match="expected_recording_count"):
        ImportSessionCreateRequest(
            robot_id="robot",
            dataset_id="dataset",
            expected_recording_count=expected_recording_count,
        )


def test_import_session_round_trips_through_json():
    session = ImportSession(
        id="session",
        org_id="org",
        robot_id="robot",
        dataset_id="dataset",
        created_by="user",
        created_at=NOW,
        expected_recording_count=3,
        recording_count=1,
        is_complete=False,
    )
    payload = json.loads(session.model_dump_json())
    assert ImportSession.model_validate(payload) == session


def test_item_keys_page_defaults_to_the_last_page():
    page = ImportSessionItemKeysPage(item_keys=["a"])
    assert page.next_cursor is None
    assert ImportSessionItemState("in_flight") == ImportSessionItemState.IN_FLIGHT


ROBOT_SOURCE = {"type": "robot_instance", "instance": 2}
IMPORT_SOURCE = {
    "type": "import_session",
    "import_session_id": "session",
    "item_key": "episodes/1.bin",
}


def test_robot_instance_start_carries_the_instance():
    request = RecordingStartRequest(**START, source=ROBOT_SOURCE)
    assert request.source == RobotInstanceSource(instance=2)


def test_a_top_level_instance_from_an_older_client_records_that_instance():
    request = RecordingStartRequest.model_validate({**START, "instance": 2})
    assert request.source == RobotInstanceSource(instance=2)
    assert "instance" not in request.model_dump()
    assert request.instance == 2


def test_instance_reads_the_robot_source_and_is_none_for_an_import():
    assert RecordingStartRequest(**START, source=ROBOT_SOURCE).instance == 2
    assert RecordingStartRequest(**START, source=IMPORT_SOURCE).instance is None


def test_import_start_carries_session_and_item_key():
    request = RecordingStartRequest(**START, source=IMPORT_SOURCE)
    assert request.source == ImportSessionSource(
        import_session_id="session", item_key="episodes/1.bin"
    )


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"instance": None},
        {"instance": -1},
        {"instance": 0, "source": IMPORT_SOURCE},
        {"instance": 2, "source": ROBOT_SOURCE},
        {"source": {"type": "robot_instance"}},
        {"source": {"type": "import_session", "import_session_id": "session"}},
        {"source": {"type": "import_session", "item_key": "a"}},
        {"source": {**IMPORT_SOURCE, "item_key": ""}},
        {
            "source": {
                **IMPORT_SOURCE,
                "item_key": "a" * (IMPORT_SESSION_ITEM_KEY_MAX_LENGTH + 1),
            }
        },
        {"source": {"type": "unknown", "instance": 2}},
    ],
)
def test_start_request_needs_exactly_one_complete_source(fields):
    with pytest.raises(ValidationError):
        RecordingStartRequest(**START, **fields)


@pytest.mark.parametrize("model", [Recording, PendingRecording])
def test_recordings_without_provenance_still_validate(model):
    payload = {"id": "rec", "org_id": "org", "progress": 0, "instance": 4}
    recording = model.model_validate(payload)
    assert recording.source is None
    assert recording.instance == 4


def test_a_robot_recording_mirrors_its_source_instance():
    recording = Recording(id="rec", org_id="org", source=ROBOT_SOURCE)
    assert recording.instance == 2
    assert recording.model_dump(mode="json")["instance"] == 2
    with pytest.raises(ValidationError, match="does not match"):
        Recording(id="rec", org_id="org", instance=3, source=ROBOT_SOURCE)


def test_an_imported_recording_has_no_robot_instance():
    recording = Recording(id="rec", org_id="org", source=IMPORT_SOURCE)
    assert recording.instance is None
    with pytest.raises(ValidationError, match="no robot instance"):
        Recording(id="rec", org_id="org", instance=0, source=IMPORT_SOURCE)


def test_an_imported_recording_leaves_instance_out_of_its_json():
    """Clients from before import sessions require instance and default it, so
    a null would fail their validation of the whole recording list."""
    recording = Recording(id="rec", org_id="org", source=IMPORT_SOURCE)
    payload = json.loads(recording.model_dump_json())
    assert "instance" not in payload
    assert payload["source"] == IMPORT_SOURCE
    assert Recording.model_validate(payload) == recording


def test_pending_recording_carries_provenance_into_a_recording():
    pending = PendingRecording(
        id="rec", org_id="org", progress=100, source=IMPORT_SOURCE
    )
    recording = Recording.model_validate(
        pending.model_dump(include=set(Recording.model_fields))
    )
    assert recording.source == ImportSessionSource(
        import_session_id="session", item_key="episodes/1.bin"
    )
    assert recording.instance is None
