"""Import sessions: a dataset import that records from a reserved instance range.

An import session owns ``IMPORT_SESSION_STRIDE`` consecutive robot instances
starting at ``IMPORT_SESSION_RESERVED_FLOOR + range_index * IMPORT_SESSION_STRIDE``.
Worker ``n`` of a session records as instance ``instance_start + n``. Instances
at or above the floor are never robot instances, so the backend resolves the
session that owns a recording from its instance alone.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, NonNegativeInt, field_validator

IMPORT_SESSION_RESERVED_FLOOR = 2**40
IMPORT_SESSION_STRIDE = 1024


class ImportSessionStatus(str, Enum):
    """Lifecycle of an import session.

    A session is only usable while active and its heartbeat is fresh. Closing it
    is final.
    """

    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ImportSessionCreateRequest(BaseModel):
    """Request to start an import into a dataset for a robot."""

    robot_id: str
    dataset_id: str
    worker_count: int = Field(ge=1, le=IMPORT_SESSION_STRIDE)


class ImportSessionCloseRequest(BaseModel):
    """Request to close an import session as completed or cancelled."""

    status: ImportSessionStatus

    @field_validator("status")
    @classmethod
    def _must_close(cls, status: ImportSessionStatus) -> ImportSessionStatus:
        if status == ImportSessionStatus.ACTIVE:
            raise ValueError(
                "an import session can only close as completed or cancelled"
            )
        return status


class ImportSession(BaseModel):
    """An import session and the instance range its workers record from.

    Attributes:
        heartbeat_timeout_s: Seconds without a heartbeat after which the session
            no longer accepts new recordings.
    """

    id: str
    org_id: str
    robot_id: str
    dataset_id: str
    created_by: str
    range_index: NonNegativeInt
    instance_start: NonNegativeInt
    worker_count: int
    status: ImportSessionStatus
    created_at: datetime
    last_heartbeat_at: datetime
    closed_at: datetime | None = None
    heartbeat_timeout_s: float
