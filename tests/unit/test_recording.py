"""Tests for the Recording model."""

import pytest
from pydantic import ValidationError

from neuracore_types import DataType, Recording


def test_recording_defaults_trace_json_size_bytes_to_empty():
    recording = Recording(id="rec", org_id="org")
    assert recording.trace_json_size_bytes == {}


def test_recording_keeps_trace_json_size_bytes_per_sensor():
    recording = Recording(
        id="rec",
        org_id="org",
        data_types={DataType.JOINT_POSITIONS},
        sensor_manifest={DataType.JOINT_POSITIONS: ["arm"]},
        trace_json_size_bytes={DataType.JOINT_POSITIONS: {"arm": 1234}},
    )
    assert recording.trace_json_size_bytes == {DataType.JOINT_POSITIONS: {"arm": 1234}}


def test_recording_rejects_trace_json_size_bytes_for_absent_data_types():
    with pytest.raises(ValidationError, match="trace_json_size_bytes"):
        Recording(
            id="rec",
            org_id="org",
            data_types={DataType.JOINT_POSITIONS},
            trace_json_size_bytes={DataType.RGB_IMAGES: {"cam0": 1234}},
        )


def test_recording_rejects_negative_trace_json_size_bytes():
    with pytest.raises(ValidationError):
        Recording(
            id="rec",
            org_id="org",
            data_types={DataType.JOINT_POSITIONS},
            trace_json_size_bytes={DataType.JOINT_POSITIONS: {"arm": -1}},
        )
