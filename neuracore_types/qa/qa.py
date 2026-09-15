"""Result types for recording QA checks."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from neuracore_types.nc_data import DataType
from neuracore_types.utils.pydantic_to_ts import (
    REQUIRED_WITH_DEFAULT_FLAG,
    fix_required_with_defaults,
)


class QAFailureReason(str, Enum):
    """Reasons why a recording can fail QA."""

    LARGE_GAPS = "LARGE_GAPS"
    INCONSISTENT_START_TIME = "INCONSISTENT_START_TIME"
    INCONSISTENT_END_TIME = "INCONSISTENT_END_TIME"
    TOO_FEW_POINTS = "TOO_FEW_POINTS"
    LOW_COVERAGE = "LOW_COVERAGE"
    STILL_LEADING = "STILL_LEADING"
    STILL_MID = "STILL_MID"
    STILL_TRAILING = "STILL_TRAILING"
    NEVER_MOVED = "NEVER_MOVED"
    TRACKING_ERROR = "TRACKING_ERROR"


class TraceIdentifier(BaseModel):
    """Identifies a single raw trace within a recording."""

    data_type: DataType
    sensor_name: str

    model_config = ConfigDict(frozen=True, json_schema_extra=fix_required_with_defaults)


class QAFinding(BaseModel):
    """One QA failure: why it failed, where in time, and optionally which trace.

    High-level pass/fail lives on the recording as ``QA_FLAGGED`` vs ``NORMAL``.
    This is the evidence behind a flag. ``start_time`` / ``end_time`` are the
    failing interval when the check has one. ``trace`` is unset for
    recording-wide findings.
    """

    reason: QAFailureReason
    start_time: float | None = Field(
        default=None, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    end_time: float | None = Field(
        default=None, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    trace: TraceIdentifier | None = Field(
        default=None, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )

    model_config = ConfigDict(frozen=True, json_schema_extra=fix_required_with_defaults)
