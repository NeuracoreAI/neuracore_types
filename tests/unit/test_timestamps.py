"""Tests for integer tick timestamps and the models that carry them."""

import re
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from pydantic import ValidationError

from neuracore_types import (
    TICKS_PER_SECOND,
    JointData,
    NCData,
    Recording,
    RecordingStartRequest,
    RecordingStopRequest,
    SynchronizedEpisode,
    SynchronizedPoint,
    convert_legacy_trace_entries,
    now_ticks,
    seconds_to_ticks,
    timestamp_to_ticks,
)


class TestTimestampToTicks:
    def test_int_is_ticks(self):
        assert timestamp_to_ticks(1_999) == 1_999

    def test_float_is_seconds(self):
        assert timestamp_to_ticks(1.5) == 1_500_000

    def test_whole_number_float_is_seconds(self):
        assert timestamp_to_ticks(2.0) == 2_000_000

    def test_numpy_integer_is_ticks(self):
        assert timestamp_to_ticks(np.int64(5)) == 5

    def test_numpy_float_is_seconds(self):
        assert timestamp_to_ticks(np.float64(1.5)) == 1_500_000

    def test_value_below_two_to_the_53_is_accepted(self):
        assert timestamp_to_ticks(2**53 - 1) == 2**53 - 1

    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    def test_non_finite_float_raises_value_error(self, value):
        with pytest.raises(ValueError):
            timestamp_to_ticks(value)

    @pytest.mark.parametrize("value", [True, "1"])
    def test_bool_and_string_raise_type_error(self, value):
        with pytest.raises(TypeError):
            timestamp_to_ticks(value)

    def test_value_at_two_to_the_53_raises_value_error(self):
        with pytest.raises(ValueError):
            timestamp_to_ticks(2**53)


def test_now_ticks_is_monotonic_nanoseconds_floored_to_ticks():
    with patch("time.monotonic_ns", return_value=123_456_789):
        ticks = now_ticks()
    assert type(ticks) is int
    assert ticks == 123_456_789 // 1000


class TestSecondsToTicks:
    def test_rounds_to_nearest_tick(self):
        assert seconds_to_ticks(0.1) == 100_000

    def test_infinite_product_raises_value_error(self):
        with pytest.raises(ValueError):
            seconds_to_ticks(sys.float_info.max)


class TestConvertLegacyTraceEntries:
    def test_bumps_collisions_without_going_backwards(self):
        entries = [
            {"timestamp": seconds} for seconds in [1.0, 1.0000004, 1.0000005, 1.0000006]
        ]
        converted = convert_legacy_trace_entries(entries)
        assert [entry["timestamp"] for entry in converted] == [
            1_000_000,
            1_000_001,
            1_000_002,
            1_000_003,
        ]

    def test_integer_entries_come_back_equal(self):
        entries = [{"timestamp": 1_000_000}, {"timestamp": 1_000_005}]
        assert convert_legacy_trace_entries(entries) == entries

    def test_second_run_matches_first(self):
        entries = [{"timestamp": 1.0}, {"timestamp": 1.0}, {"timestamp": 2.5}]
        once = convert_legacy_trace_entries(entries)
        assert convert_legacy_trace_entries(once) == once

    def test_other_keys_are_kept(self):
        entries = [{"timestamp": 1.0, "frame_idx": 0, "value": 0.5}]
        assert convert_legacy_trace_entries(entries) == [
            {"timestamp": 1_000_000, "frame_idx": 0, "value": 0.5}
        ]


@pytest.mark.parametrize("model", [NCData, SynchronizedPoint])
class TestModelTimestamp:
    def test_int_is_kept(self, model):
        assert model(timestamp=1).timestamp == 1

    def test_float_is_converted(self, model):
        assert model(timestamp=1.5).timestamp == 1_500_000

    def test_json_float_is_converted(self, model):
        assert model.model_validate_json('{"timestamp": 1.0}').timestamp == 1_000_000

    @pytest.mark.parametrize("value", [True, "1"])
    def test_bool_and_string_are_rejected(self, model, value):
        with pytest.raises(ValidationError):
            model(timestamp=value)

    def test_default_is_int(self, model):
        assert type(model().timestamp) is int


def test_subclass_timestamp_is_converted():
    assert JointData(value=0.1, timestamp=2.5).timestamp == 2_500_000


def test_synchronized_episode_validates_with_ticks_and_seconds():
    episode = SynchronizedEpisode(
        observations=[SynchronizedPoint(timestamp=1_000_000)],
        start_timestamp=1_000_000,
        end_timestamp=2_000_000,
        ticks_per_second=TICKS_PER_SECOND,
        start_time=1.0,
        end_time=2.0,
        robot_id="robot",
    )
    assert episode.start_timestamp == 1_000_000
    assert episode.end_timestamp == 2_000_000
    assert episode.start_time == 1.0


@pytest.mark.parametrize("field", ["start_timestamp", "end_timestamp"])
def test_synchronized_episode_window_float_is_converted(field):
    fields = {
        "observations": [],
        "start_timestamp": 1_000_000,
        "end_timestamp": 2_000_000,
        "ticks_per_second": TICKS_PER_SECOND,
        "start_time": 1.0,
        "end_time": 2.0,
        "robot_id": "robot",
    }
    episode = SynchronizedEpisode(**{**fields, field: 1.5})
    assert getattr(episode, field) == 1_500_000


@pytest.mark.parametrize("field", ["start_timestamp", "end_timestamp"])
def test_recording_window_float_is_converted(field):
    recording = Recording(id="recording", org_id="org", **{field: 1.5})
    assert getattr(recording, field) == 1_500_000


def test_synchronized_episode_requires_tick_fields():
    with pytest.raises(ValidationError):
        SynchronizedEpisode(
            observations=[], start_time=1.0, end_time=2.0, robot_id="robot"
        )


def test_recording_without_tick_fields_validates_with_none():
    recording = Recording(id="recording", org_id="org")
    assert recording.ticks_per_second is None
    assert recording.start_timestamp is None
    assert recording.end_timestamp is None


def test_recording_start_request_without_tick_fields_validates_with_none():
    request = RecordingStartRequest(
        robot_id="robot", instance=0, dataset_id="dataset", start_time=1.0
    )
    assert request.ticks_per_second is None
    assert request.start_timestamp is None


def test_recording_start_request_float_start_timestamp_is_converted():
    request = RecordingStartRequest(
        robot_id="robot",
        instance=0,
        dataset_id="dataset",
        start_time=1.0,
        start_timestamp=1.5,
    )
    assert request.start_timestamp == 1_500_000


def test_recording_start_request_bool_start_timestamp_is_rejected():
    with pytest.raises(ValidationError):
        RecordingStartRequest(
            robot_id="robot",
            instance=0,
            dataset_id="dataset",
            start_time=1.0,
            start_timestamp=True,
        )


def test_recording_stop_request_float_end_timestamp_is_converted():
    request = RecordingStopRequest(
        recording_id="recording", end_time=2.0, end_timestamp=1.5
    )
    assert request.end_timestamp == 1_500_000


def test_recording_stop_request_bool_end_timestamp_is_rejected():
    with pytest.raises(ValidationError):
        RecordingStopRequest(recording_id="recording", end_time=2.0, end_timestamp=True)


def test_recording_stop_request_without_tick_fields_validates_with_none():
    request = RecordingStopRequest(recording_id="recording", end_time=2.0)
    assert request.end_timestamp is None


def test_typescript_constant_matches_python_constant():
    constants_ts = Path(__file__).parents[2] / "neuracore_types" / "constants.ts"
    match = re.search(
        r"export const TICKS_PER_SECOND = ([\d_]+);", constants_ts.read_text()
    )
    assert match is not None
    assert int(match.group(1)) == TICKS_PER_SECOND
