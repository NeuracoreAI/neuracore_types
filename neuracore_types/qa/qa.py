"""Result types for recording QA checks."""

from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from neuracore_types.utils.pydantic_to_ts import (
    REQUIRED_WITH_DEFAULT_FLAG,
    fix_required_with_defaults,
)


class QAFailureReason(str, Enum):
    """Reasons why a recording can fail QA."""

    LARGE_GAPS = "LARGE_GAPS"
    INCONSISTENT_START_TIME = "INCONSISTENT_START_TIME"


class QAPassResult(BaseModel):
    """Result when a recording passes a QA check."""

    passed: Literal[True] = Field(
        default=True, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )

    model_config = ConfigDict(frozen=True, json_schema_extra=fix_required_with_defaults)


class QAFailureResult(BaseModel):
    """Result when a recording fails a QA check."""

    passed: Literal[False] = Field(
        default=False, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    reason: QAFailureReason

    model_config = ConfigDict(frozen=True, json_schema_extra=fix_required_with_defaults)


QACheckResult = Annotated[
    Union[QAPassResult, QAFailureResult], Field(discriminator="passed")
]
