"""Tests for the tags of RecordingStopRequest."""

import pytest
from pydantic import ValidationError

from neuracore_types import RecordingStopRequest


def _request(**fields: object) -> RecordingStopRequest:
    return RecordingStopRequest(recording_id="rec-1", end_time=1.0, **fields)


def test_request_without_tags_has_none() -> None:
    assert _request().tags is None


def test_tags_are_normalized_and_deduplicated() -> None:
    request = _request(tags=["  Good Run ", "good run", "Edge"])

    assert request.tags == ["good-run", "edge"]


def test_qa_failure_reason_name_becomes_a_user_name() -> None:
    assert _request(tags=["NEVER_MOVED"]).tags == ["never_moved"]


def test_invalid_tag_name_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _request(tags=["a,b"])
