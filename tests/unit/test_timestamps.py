"""Tests for microsecond timestamps and the models that carry them."""

import re
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from neuracore_types import (
    MICROSECONDS_PER_SECOND,
    TIMESTAMP_US_LIMIT,
    JointData,
    NCData,
    Recording,
    RecordingStartRequest,
    RecordingStopRequest,
    SynchronizedEpisode,
    SynchronizedPoint,
    add_microsecond_timestamps,
    now_us,
    seconds_to_us,
)

CONSTANTS_TS = Path(__file__).parents[2] / "neuracore_types" / "constants.ts"
EPOCH_SECONDS = 1_790_000_000.25
EPOCH_US = 1_790_000_000_250_000


def test_now_us_is_wall_clock_nanoseconds_floored_to_microseconds():
    with patch("time.time_ns", return_value=1_790_000_000_123_456_789):
        assert now_us() == 1_790_000_000_123_456


class TestSecondsToUs:
    def test_rounds_to_nearest_microsecond(self):
        assert seconds_to_us(1.0000004) == 1_000_000
        assert seconds_to_us(1.0000006) == 1_000_001

    def test_epoch_value_is_exact(self):
        assert seconds_to_us(EPOCH_SECONDS) == EPOCH_US

    @pytest.mark.parametrize("seconds", [float("nan"), float("inf"), 1e308])
    def test_non_finite_microsecond_value_raises_value_error(self, seconds):
        with pytest.raises(ValueError):
            seconds_to_us(seconds)


class TestAddMicrosecondTimestamps:
    def test_rounding_collisions_are_bumped_to_stay_increasing(self):
        entries = [
            {"timestamp": seconds} for seconds in [1.0, 1.0000004, 1.0000005, 1.0000006]
        ]
        converted = add_microsecond_timestamps(entries)
        assert [entry["timestamp_us"] for entry in converted] == [
            1_000_000,
            1_000_001,
            1_000_002,
            1_000_003,
        ]

    def test_second_run_changes_nothing(self):
        entries = [{"timestamp": 1.0}, {"timestamp": 1.0}, {"timestamp": 2.5}]
        converted = add_microsecond_timestamps(entries)
        assert add_microsecond_timestamps(converted) == converted

    def test_entry_with_timestamp_us_is_untouched_when_timestamp_disagrees(self):
        entry = {"timestamp": 9.0, "timestamp_us": 1_000_000}
        assert add_microsecond_timestamps([entry]) == [
            {"timestamp": 9.0, "timestamp_us": 1_000_000}
        ]

    def test_other_keys_are_kept(self):
        entries = [{"timestamp": 1.0, "frame_idx": 0, "offset": 4, "length": 8}]
        assert add_microsecond_timestamps(entries) == [{
            "timestamp": 1.0,
            "frame_idx": 0,
            "offset": 4,
            "length": 8,
            "timestamp_us": 1_000_000,
        }]

    def test_input_entries_are_not_changed(self):
        entries = [{"timestamp": 1.0}]
        add_microsecond_timestamps(entries)
        assert entries == [{"timestamp": 1.0}]


@pytest.mark.parametrize(
    ("model", "fields"),
    [(NCData, {}), (JointData, {"value": 0.5}), (SynchronizedPoint, {})],
    ids=["NCData", "JointData", "SynchronizedPoint"],
)
class TestModelTimestamps:
    def test_timestamp_us_given_recomputes_timestamp(self, model, fields):
        instance = model(**fields, timestamp=9.0, timestamp_us=EPOCH_US)
        assert instance.timestamp_us == EPOCH_US
        assert instance.timestamp == EPOCH_SECONDS

    def test_only_timestamp_given_rounds_timestamp_us(self, model, fields):
        instance = model(**fields, timestamp=1.0000006)
        assert instance.timestamp_us == 1_000_001
        assert instance.timestamp == 1.000001

    def test_neither_given_defaults_to_the_wall_clock(self, model, fields):
        with patch("time.time", return_value=EPOCH_SECONDS):
            instance = model(**fields)
        assert instance.timestamp_us == EPOCH_US
        assert instance.timestamp == EPOCH_SECONDS

    def test_model_dump_carries_both_fields(self, model, fields):
        dumped = model(**fields, timestamp_us=EPOCH_US).model_dump(mode="json")
        assert dumped["timestamp_us"] == EPOCH_US
        assert dumped["timestamp"] == EPOCH_SECONDS

    def test_json_round_trip_keeps_timestamp_us(self, model, fields):
        instance = model(**fields, timestamp_us=EPOCH_US)
        loaded = model.model_validate_json(instance.model_dump_json())
        assert loaded.timestamp_us == EPOCH_US

    @pytest.mark.parametrize("value", [True, "1", 1.0, TIMESTAMP_US_LIMIT, -1])
    def test_invalid_timestamp_us_is_rejected(self, model, fields, value):
        with pytest.raises(ValidationError):
            model(**fields, timestamp_us=value)

    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    def test_non_finite_timestamp_is_rejected(self, model, fields, value):
        with pytest.raises(ValidationError):
            model(**fields, timestamp=value)

    @pytest.mark.parametrize("value", [-1.0, 1e300, 1e303])
    def test_timestamp_outside_microsecond_bounds_is_rejected(
        self, model, fields, value
    ):
        with pytest.raises(ValidationError):
            model(**fields, timestamp=value)

    def test_model_construct_derives_timestamp_us(self, model, fields):
        instance = model.model_construct(**fields, timestamp=EPOCH_SECONDS)
        assert instance.timestamp_us == EPOCH_US


def _episode(**fields: object) -> SynchronizedEpisode:
    return SynchronizedEpisode(
        observations=[SynchronizedPoint(timestamp=1.0)], robot_id="robot", **fields
    )


class TestSynchronizedEpisodeWindow:
    def test_window_without_microseconds_is_derived(self):
        episode = _episode(start_time=1.0, end_time=2.5)
        assert episode.start_timestamp_us == 1_000_000
        assert episode.end_timestamp_us == 2_500_000

    def test_window_microseconds_recompute_seconds(self):
        episode = _episode(
            start_time=9.0,
            end_time=9.0,
            start_timestamp_us=1_000_000,
            end_timestamp_us=2_500_000,
        )
        assert episode.start_time == 1.0
        assert episode.end_time == 2.5

    def test_order_keeps_window_microseconds(self):
        episode = _episode(
            start_time=1.0,
            end_time=2.0,
            start_timestamp_us=EPOCH_US,
            end_timestamp_us=EPOCH_US + 1,
        )
        ordered = episode.order({})
        assert ordered.start_timestamp_us == EPOCH_US
        assert ordered.end_timestamp_us == EPOCH_US + 1
        assert ordered.observations[0].timestamp_us == 1_000_000

    @pytest.mark.parametrize("field", ["start_time", "end_time"])
    def test_non_finite_window_is_rejected(self, field):
        with pytest.raises(ValidationError):
            _episode(**{"start_time": 1.0, "end_time": 2.0, field: float("nan")})

    def test_missing_start_time_is_a_validation_error(self):
        with pytest.raises(ValidationError, match="start_time"):
            _episode(end_time=2.0)


class TestRecordingWindow:
    def test_recording_without_window_has_none(self):
        recording = Recording(id="recording", org_id="org")
        assert recording.start_timestamp_us is None
        assert recording.end_timestamp_us is None

    def test_recording_keeps_window(self):
        recording = Recording(
            id="recording",
            org_id="org",
            start_timestamp_us=EPOCH_US,
            end_timestamp_us=EPOCH_US + 1,
        )
        assert recording.start_timestamp_us == EPOCH_US
        assert recording.end_timestamp_us == EPOCH_US + 1

    def test_start_request_carries_optional_start_timestamp_us(self):
        fields = {
            "robot_id": "robot",
            "instance": 0,
            "dataset_id": "dataset",
            "start_time": EPOCH_SECONDS,
        }
        assert RecordingStartRequest(**fields).start_timestamp_us is None
        request = RecordingStartRequest(**fields, start_timestamp_us=EPOCH_US)
        assert request.start_timestamp_us == EPOCH_US

    def test_stop_request_carries_optional_end_timestamp_us(self):
        fields = {"recording_id": "recording", "end_time": EPOCH_SECONDS}
        assert RecordingStopRequest(**fields).end_timestamp_us is None
        request = RecordingStopRequest(**fields, end_timestamp_us=EPOCH_US)
        assert request.end_timestamp_us == EPOCH_US

    def test_float_window_is_rejected(self):
        with pytest.raises(ValidationError):
            Recording(id="recording", org_id="org", start_timestamp_us=EPOCH_SECONDS)
        with pytest.raises(ValidationError):
            RecordingStartRequest(
                robot_id="robot",
                instance=0,
                dataset_id="dataset",
                start_time=EPOCH_SECONDS,
                start_timestamp_us=1_790_000_000.0,
            )
        with pytest.raises(ValidationError):
            RecordingStopRequest(
                recording_id="recording",
                end_time=EPOCH_SECONDS,
                end_timestamp_us=1_790_000_000.0,
            )


def test_typescript_constant_matches_python_constant():
    match = re.search(
        r"export const MICROSECONDS_PER_SECOND = ([\d_]+);", CONSTANTS_TS.read_text()
    )
    assert match is not None
    assert int(match.group(1)) == MICROSECONDS_PER_SECOND
