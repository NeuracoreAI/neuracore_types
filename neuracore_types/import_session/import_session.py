"""Import sessions: one dataset import and the recordings it selected.

Each imported recording carries an ``item_key`` that names its source item and
stays the same when the import is resumed. A session saves at most one
recording per item key, and it is complete once every selected item has been
saved.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, NonNegativeInt, PositiveInt

# Item keys are primary key values; 512 characters keeps a UTF-8 key inside a
# PostgreSQL B-tree entry.
IMPORT_SESSION_ITEM_KEY_MAX_LENGTH = 512


class ImportSessionCreateRequest(BaseModel):
    """Request to start an import of selected items into a dataset for a robot.

    Attributes:
        expected_recording_count: How many items the import selected. A resumed
            import keeps the session and the count it was created with.
    """

    robot_id: str
    dataset_id: str
    expected_recording_count: PositiveInt


class ImportSession(BaseModel):
    """An import session and how far its recordings have been saved.

    Attributes:
        recording_count: Distinct item keys with a saved, undeleted recording.
        is_complete: Whether recording_count has reached expected_recording_count.
    """

    id: str
    org_id: str
    robot_id: str
    dataset_id: str
    created_by: str
    created_at: datetime
    expected_recording_count: PositiveInt
    recording_count: NonNegativeInt
    is_complete: bool


class ImportSessionItemState(str, Enum):
    """Which item keys of an import session to list.

    SAVED keys have a saved recording. IN_FLIGHT keys have a stopped recording
    that is still uploading, so sending the item again is not needed yet.
    """

    SAVED = "saved"
    IN_FLIGHT = "in_flight"


class ImportSessionItemKeysPage(BaseModel):
    """A page of an import session's item keys in ascending order.

    Attributes:
        next_cursor: Pass back as ``cursor`` for the next page; None on the
            last page.
    """

    item_keys: list[str]
    next_cursor: str | None = None
