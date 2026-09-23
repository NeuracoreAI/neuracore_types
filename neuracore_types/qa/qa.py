"""Result types for recording QA checks."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, computed_field

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


QA_FAILURE_REASON_DESCRIPTIONS: dict[QAFailureReason, str] = {
    QAFailureReason.LARGE_GAPS: "Large gaps in data",
    QAFailureReason.INCONSISTENT_START_TIME: "Inconsistent start time",
    QAFailureReason.INCONSISTENT_END_TIME: "Inconsistent end time",
    QAFailureReason.TOO_FEW_POINTS: "Too few data points",
    QAFailureReason.LOW_COVERAGE: "Low data coverage",
    QAFailureReason.STILL_LEADING: "Stationary at start",
    QAFailureReason.STILL_MID: "Stationary mid-recording",
    QAFailureReason.STILL_TRAILING: "Stationary at end",
    QAFailureReason.NEVER_MOVED: "No movement detected",
    QAFailureReason.TRACKING_ERROR: "Tracking error",
}


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

    @computed_field  # type: ignore[prop-decorator]
    @property
    def description(self) -> str:
        """Human-readable description of the failure reason."""
        return QA_FAILURE_REASON_DESCRIPTIONS[self.reason]

    model_config = ConfigDict(frozen=True, json_schema_extra=fix_required_with_defaults)
