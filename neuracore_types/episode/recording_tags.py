"""Tags that users and QA attach to a recording in a dataset."""

import re
import unicodedata
from enum import Enum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)
from pydantic.config import JsonDict

from neuracore_types.qa.qa import (
    QA_FAILURE_REASON_DESCRIPTIONS,
    QAFailureReason,
    QAFinding,
)
from neuracore_types.utils.pydantic_to_ts import (
    REQUIRED_WITH_DEFAULT_FLAG,
    fix_required_with_defaults,
)

MAX_TAG_NAME_LENGTH = 50
FLAGGED_TAG_NAME = "flagged"

_WHITESPACE_RUN = re.compile(r"\s+")
_TAG_NAME_SEPARATOR = "-"
_FORBIDDEN_CHARACTERS = {","}
_CONTROL_CHARACTER_CATEGORY = "Cc"
_QA_TAG_NAMES = {reason.value for reason in QAFailureReason}
_QA_TAG_RANK = {reason.value: rank for rank, reason in enumerate(QAFailureReason)}
_QA_TAG_DESCRIPTIONS = {
    reason.value: description
    for reason, description in QA_FAILURE_REASON_DESCRIPTIONS.items()
}


def is_qa_tag_name(name: str) -> bool:
    """Return whether a tag name is a QA tag name.

    QA tag names are QAFailureReason values. User tag names are lower case,
    so the two sets of names never meet.

    Args:
        name: Tag name.

    Returns:
        True when the name is a QAFailureReason value.
    """
    return name in _QA_TAG_NAMES


def normalize_user_tag_name(raw: str) -> str:
    """Return the normalized form of a user tag name.

    The name is stripped, made lower case, and each run of white space
    becomes one hyphen.

    Args:
        raw: Tag name as the user typed it.

    Returns:
        The normalized tag name.

    Raises:
        ValueError: If the normalized name is empty, longer than
            MAX_TAG_NAME_LENGTH, or holds a control character or a comma.
    """
    name = _WHITESPACE_RUN.sub(_TAG_NAME_SEPARATOR, raw.strip().lower())
    if not name:
        raise ValueError("Tag name must not be empty")
    if len(name) > MAX_TAG_NAME_LENGTH:
        raise ValueError(
            f"Tag name must be at most {MAX_TAG_NAME_LENGTH} characters long"
        )
    if any(
        character in _FORBIDDEN_CHARACTERS
        or unicodedata.category(character) == _CONTROL_CHARACTER_CATEGORY
        for character in name
    ):
        raise ValueError("Tag name must not hold a comma or a control character")
    return name


class RecordingTagSource(str, Enum):
    """Who wrote a recording tag.

    Attributes:
        USER: A user added the tag.
        QA: A QA check found a failure. Users cannot change these tags.
    """

    USER = "USER"
    QA = "QA"


class RecordingTag(BaseModel):
    """A tag on a recording in a dataset.

    A user tag has a lower case name and no findings. A QA tag is named
    after one QAFailureReason and holds the findings of that reason.

    Attributes:
        name: Name of the tag. It identifies the tag.
        source: Who wrote the tag.
        findings: QA findings behind the tag. Empty for a user tag.
    """

    name: str
    source: RecordingTagSource
    findings: list[QAFinding] = Field(
        default_factory=list, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )

    model_config = ConfigDict(frozen=True, json_schema_extra=fix_required_with_defaults)

    @model_validator(mode="after")
    def _check_source_rules(self) -> "RecordingTag":
        if self.source == RecordingTagSource.USER:
            self._check_user_tag()
        else:
            self._check_qa_tag()
        return self

    def _check_user_tag(self) -> None:
        if self.name != normalize_user_tag_name(self.name):
            raise ValueError(f"User tag name {self.name!r} is not normalized")
        if self.findings:
            raise ValueError("A user tag must not have findings")

    def _check_qa_tag(self) -> None:
        if not is_qa_tag_name(self.name):
            raise ValueError(f"QA tag name {self.name!r} is not a QA failure reason")
        if not self.findings:
            raise ValueError("A QA tag must have at least one finding")
        if any(finding.reason.value != self.name for finding in self.findings):
            raise ValueError(
                f"Every finding of QA tag {self.name} must have its reason"
            )


def _keep_only_ref(json_schema: JsonDict) -> None:
    """Remove the keys beside ``$ref`` from a property schema.

    The TypeScript generator makes a second copy of an enum for a ``$ref``
    that has other keys, such as the ``readOnly`` of a computed field.
    """
    for key in [key for key in json_schema if key != "$ref"]:
        del json_schema[key]


class RecordingTagCount(BaseModel):
    """Number of recordings that have a tag.

    Attributes:
        name: Name of the tag.
        recording_count: Number of recordings with the tag.
        source: Who writes tags of this name. Input values are ignored.
        description: Display text of a QA tag. None for a user tag.
    """

    name: str
    recording_count: int

    @computed_field(json_schema_extra=_keep_only_ref)  # type: ignore[prop-decorator]
    @property
    def source(self) -> RecordingTagSource:
        """Who writes tags of this name."""
        if is_qa_tag_name(self.name):
            return RecordingTagSource.QA
        return RecordingTagSource.USER

    @computed_field  # type: ignore[prop-decorator]
    @property
    def description(self) -> str | None:
        """Display text of a QA tag. None for a user tag."""
        return _QA_TAG_DESCRIPTIONS.get(self.name)


class RecordingTagRequest(BaseModel):
    """Request to add or remove a user tag.

    Attributes:
        name: Tag name. It is normalized on validation.
    """

    name: str

    @field_validator("name")
    @classmethod
    def _normalize_name(cls, name: str) -> str:
        return normalize_user_tag_name(name)


def qa_tags_from_findings(findings: list[QAFinding]) -> list[RecordingTag]:
    """Group QA findings into one QA tag per failure reason.

    Args:
        findings: QA findings of one recording.

    Returns:
        The QA tags in QAFailureReason declaration order. Each tag keeps its
        findings in input order.
    """
    findings_by_reason: dict[QAFailureReason, list[QAFinding]] = {
        reason: [] for reason in QAFailureReason
    }
    for finding in findings:
        findings_by_reason[finding.reason].append(finding)
    return [
        RecordingTag(
            name=reason.value, source=RecordingTagSource.QA, findings=reason_findings
        )
        for reason, reason_findings in findings_by_reason.items()
        if reason_findings
    ]


def to_stored_tags(tags: list[RecordingTag]) -> list[RecordingTag]:
    """Return the tags in the order a recording stores them.

    QA tags come first in QAFailureReason order, then user tags in their
    given order.

    Args:
        tags: Tags of one recording.

    Returns:
        The tags in stored order.

    Raises:
        ValueError: If two tags have the same name.
    """
    names = [tag.name for tag in tags]
    if len(names) != len(set(names)):
        raise ValueError("Tag names of a recording must be unique")
    qa_tags = [tag for tag in tags if tag.source == RecordingTagSource.QA]
    user_tags = [tag for tag in tags if tag.source == RecordingTagSource.USER]
    return [*sorted(qa_tags, key=lambda tag: _QA_TAG_RANK[tag.name]), *user_tags]
