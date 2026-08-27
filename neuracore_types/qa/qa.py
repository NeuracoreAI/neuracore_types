"""Result types for recording QA checks."""

from enum import Enum
from typing import Annotated, Literal, Union

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


class TraceIdentifier(BaseModel):
    """Identifies a single raw trace within a recording."""

    data_type: DataType
    sensor_name: str

    model_config = ConfigDict(frozen=True, json_schema_extra=fix_required_with_defaults)


class QAPassResult(BaseModel):
    """Result when a recording passes a QA check."""

    passed: Literal[True] = Field(
        default=True, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )

    model_config = ConfigDict(frozen=True, json_schema_extra=fix_required_with_defaults)


class QAFailureResult(BaseModel):
    """Result when a recording fails a QA check.

    ``affected_traces`` names the specific traces the failure was found on.
    An empty list means the failure applies to the recording as a whole,
    rather than to any particular trace.
    """

    passed: Literal[False] = Field(
        default=False, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    reason: QAFailureReason
    affected_traces: list[TraceIdentifier] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True, json_schema_extra=fix_required_with_defaults)


QACheckResult = Annotated[
    Union[QAPassResult, QAFailureResult], Field(discriminator="passed")
]
