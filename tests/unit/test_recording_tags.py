"""Tests for recording tags."""

import re

import pytest
from pydantic import ValidationError

from neuracore_types import (
    FLAGGED_TAG_NAME,
    MAX_TAG_NAME_LENGTH,
    QAFailureReason,
    QAFinding,
    RecordingMetadata,
    RecordingSort,
    RecordingSortField,
    RecordingTag,
    RecordingTagCount,
    RecordingTagRequest,
    RecordingTagSource,
    is_qa_tag_name,
    normalize_user_tag_name,
    qa_tags_from_findings,
)

STILL_MID_EARLY = QAFinding(reason=QAFailureReason.STILL_MID, start_time=1.0)
STILL_MID_LATE = QAFinding(reason=QAFailureReason.STILL_MID, start_time=5.0)
LARGE_GAPS = QAFinding(reason=QAFailureReason.LARGE_GAPS, start_time=3.0)
NEVER_MOVED = QAFinding(reason=QAFailureReason.NEVER_MOVED)

FLAGGED_TAG = RecordingTag(name=FLAGGED_TAG_NAME, source=RecordingTagSource.USER)
REVIEW_TAG = RecordingTag(name="needs-review", source=RecordingTagSource.USER)
ZETA_TAG = RecordingTag(name="zeta", source=RecordingTagSource.USER)
ALPHA_TAG = RecordingTag(name="alpha", source=RecordingTagSource.USER)


def qa_tag(reason, *findings):
    return RecordingTag(
        name=reason.value, source=RecordingTagSource.QA, findings=list(findings)
    )


STILL_MID_TAG = qa_tag(QAFailureReason.STILL_MID, STILL_MID_EARLY, STILL_MID_LATE)
NEVER_MOVED_TAG = qa_tag(QAFailureReason.NEVER_MOVED, NEVER_MOVED)
LARGE_GAPS_TAG = qa_tag(QAFailureReason.LARGE_GAPS, LARGE_GAPS)
STILL_MID_EARLY_TAG = qa_tag(QAFailureReason.STILL_MID, STILL_MID_EARLY)

EMPTY_NAME_MESSAGE = "Tag name must not be empty"
LONG_NAME_MESSAGE = f"Tag name must be at most {MAX_TAG_NAME_LENGTH} characters long"
BAD_CHARACTER_MESSAGE = "Tag name must not hold a comma or a control character"
UNIQUE_NAMES_MESSAGE = "Tag names of a recording must be unique"
NOT_NORMALIZED_MESSAGE = "User tag name {name!r} is not normalized"
NOT_A_REASON_MESSAGE = "QA tag name {name!r} is not a QA failure reason"


def pydantic_message(message):
    return re.escape(f"Value error, {message} [")


def build_with_tags(tags):
    return RecordingMetadata(tags=tags)


def assign_tags(tags):
    metadata = RecordingMetadata()
    metadata.tags = tags
    return metadata


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  Needs  Review\tNow ", "needs-review-now"),
        ("Flagged", "flagged"),
        ("a" * MAX_TAG_NAME_LENGTH, "a" * MAX_TAG_NAME_LENGTH),
        ("a\u200db", "a\u200db"),
        ("\U0001f469\u200d\U0001f52c", "\U0001f469\u200d\U0001f52c"),
    ],
)
def test_normalize_user_tag_name_returns_normalized_form(raw, expected):
    assert normalize_user_tag_name(raw) == expected


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("", EMPTY_NAME_MESSAGE),
        (" \t\n ", EMPTY_NAME_MESSAGE),
        ("a" * (MAX_TAG_NAME_LENGTH + 1), LONG_NAME_MESSAGE),
        ("bad,tag", BAD_CHARACTER_MESSAGE),
        ("bad\x00tag", BAD_CHARACTER_MESSAGE),
        ("a\x7f", BAD_CHARACTER_MESSAGE),
    ],
)
def test_normalize_user_tag_name_rejects_invalid_names(raw, message):
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        normalize_user_tag_name(raw)


def test_tag_request_normalizes_name():
    assert RecordingTagRequest(name=" Needs Review ").name == "needs-review"


def test_tag_request_rejects_invalid_name():
    with pytest.raises(ValidationError):
        RecordingTagRequest(name="a,b")


@pytest.mark.parametrize(
    ("name", "source", "findings", "message"),
    [
        ("Needs-Review", "USER", [], NOT_NORMALIZED_MESSAGE),
        ("needs review", "USER", [], NOT_NORMALIZED_MESSAGE),
        ("NEVER_MOVED", "USER", [], NOT_NORMALIZED_MESSAGE),
        ("needs-review", "USER", [LARGE_GAPS], "A user tag must not have findings"),
        ("flagged", "QA", [LARGE_GAPS], NOT_A_REASON_MESSAGE),
        ("large_gaps", "QA", [LARGE_GAPS], NOT_A_REASON_MESSAGE),
        ("LARGE_GAPS", "QA", [], "A QA tag must have at least one finding"),
        (
            "LARGE_GAPS",
            "QA",
            [LARGE_GAPS, STILL_MID_EARLY],
            "Every finding of QA tag LARGE_GAPS must have its reason",
        ),
    ],
)
def test_tag_rejects_names_and_findings_its_source_forbids(
    name, source, findings, message
):
    expected = pydantic_message(message.format(name=name))
    with pytest.raises(ValidationError, match=expected):
        RecordingTag(name=name, source=source, findings=findings)


def test_qa_tags_from_findings_groups_by_reason_in_reason_order():
    findings = [STILL_MID_LATE, LARGE_GAPS, NEVER_MOVED, STILL_MID_EARLY]

    assert qa_tags_from_findings(findings) == [
        NEVER_MOVED_TAG,
        LARGE_GAPS_TAG,
        qa_tag(QAFailureReason.STILL_MID, STILL_MID_LATE, STILL_MID_EARLY),
    ]


def test_qa_tags_from_findings_returns_no_tags_for_no_findings():
    assert qa_tags_from_findings([]) == []


def test_metadata_defaults_to_no_tags():
    metadata = RecordingMetadata()

    assert metadata.tags == []
    assert metadata.tag_count == 0


def test_metadata_ignores_legacy_status_fields_in_input():
    metadata = RecordingMetadata.model_validate({
        "status": "FLAGGED",
        "qa_flag_reasons": [LARGE_GAPS.model_dump()],
    })

    assert metadata.tags == []
    assert "status" not in metadata.model_dump()
    assert "qa_flag_reasons" not in metadata.model_dump()


@pytest.mark.parametrize("make_metadata", [build_with_tags, assign_tags])
def test_tags_are_stored_with_qa_tags_first_in_reason_order(make_metadata):
    metadata = make_metadata([
        ZETA_TAG,
        STILL_MID_TAG,
        FLAGGED_TAG,
        ALPHA_TAG,
        LARGE_GAPS_TAG,
        NEVER_MOVED_TAG,
    ])

    assert metadata.tags == [
        NEVER_MOVED_TAG,
        LARGE_GAPS_TAG,
        STILL_MID_TAG,
        ZETA_TAG,
        FLAGGED_TAG,
        ALPHA_TAG,
    ]


def test_user_tag_added_after_qa_tags_stays_after_them():
    metadata = RecordingMetadata(tags=[LARGE_GAPS_TAG, ZETA_TAG])

    reloaded = RecordingMetadata.model_validate({
        **metadata.model_dump(),
        "tags": [*metadata.model_dump()["tags"], ALPHA_TAG.model_dump()],
    })

    assert reloaded.tags == [LARGE_GAPS_TAG, ZETA_TAG, ALPHA_TAG]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (QAFailureReason.NEVER_MOVED.value, True),
        ("never_moved", False),
        ("LARGE_GAP", False),
        ("LARGE_GAPS_X", False),
        (FLAGGED_TAG_NAME, False),
    ],
)
def test_is_qa_tag_name_matches_failure_reason_values(name, expected):
    assert is_qa_tag_name(name) is expected


SAME_NAME_TAG_PAIRS = [
    (REVIEW_TAG, REVIEW_TAG),
    (REVIEW_TAG, RecordingTag(name="needs-review", source=RecordingTagSource.USER)),
    (STILL_MID_EARLY_TAG, qa_tag(QAFailureReason.STILL_MID, STILL_MID_LATE)),
    (STILL_MID_EARLY_TAG, STILL_MID_TAG),
]


@pytest.mark.parametrize(("first", "second"), SAME_NAME_TAG_PAIRS)
def test_metadata_rejects_duplicate_tag_names(first, second):
    with pytest.raises(ValidationError, match=pydantic_message(UNIQUE_NAMES_MESSAGE)):
        RecordingMetadata(tags=[first, second])


@pytest.mark.parametrize(("first", "second"), SAME_NAME_TAG_PAIRS)
def test_assigning_duplicate_tag_names_is_rejected_and_keeps_tags(first, second):
    metadata = RecordingMetadata(tags=[LARGE_GAPS_TAG, ZETA_TAG])

    with pytest.raises(ValueError, match=re.escape(UNIQUE_NAMES_MESSAGE)):
        metadata.tags = [first, second]

    assert metadata.tags == [LARGE_GAPS_TAG, ZETA_TAG]
    assert metadata.tag_names == ["LARGE_GAPS", "zeta"]
    assert metadata.tag_count == 2


def test_metadata_dump_has_tag_names_and_tag_count_and_ignores_them_in_input():
    metadata = RecordingMetadata.model_validate({
        "tags": [tag.model_dump() for tag in (ZETA_TAG, NEVER_MOVED_TAG, ALPHA_TAG)],
        "tag_names": ["stale"],
        "tag_count": 7,
    })

    dumped = metadata.model_dump()
    assert dumped["tag_names"] == ["NEVER_MOVED", "zeta", "alpha"]
    assert dumped["tag_count"] == 3


def test_metadata_survives_a_dump_and_load_round_trip():
    metadata = RecordingMetadata(tags=[REVIEW_TAG, STILL_MID_TAG, FLAGGED_TAG])

    assert RecordingMetadata.model_validate(metadata.model_dump()) == metadata


@pytest.mark.parametrize(
    ("model", "mode", "required"),
    [
        (
            RecordingMetadata,
            "serialization",
            {"name", "notes", "tags", "tag_names", "tag_count"},
        ),
        (RecordingMetadata, "validation", {"name", "notes", "tags"}),
        (RecordingTag, "serialization", {"name", "source", "findings"}),
        (
            RecordingTagCount,
            "serialization",
            {"name", "recording_count", "source", "description"},
        ),
    ],
)
def test_schema_required_fields(model, mode, required):
    assert set(model.model_json_schema(mode=mode)["required"]) == required


def test_tag_count_schema_reuses_the_tag_source_enum():
    schema = RecordingTagCount.model_json_schema(mode="serialization")

    assert schema["properties"]["source"] == {"$ref": "#/$defs/RecordingTagSource"}


def test_recordings_can_be_sorted_by_tag_count():
    sort = RecordingSort.model_validate({"field": "tag_count"})

    assert sort.field == RecordingSortField.TAG_COUNT


def test_qa_failure_reasons_are_declared_in_display_order():
    expected = """NEVER_MOVED TRACKING_ERROR LARGE_GAPS LOW_COVERAGE TOO_FEW_POINTS
        INCONSISTENT_START_TIME INCONSISTENT_END_TIME STILL_MID STILL_LEADING
        STILL_TRAILING""".split()

    assert [reason.value for reason in QAFailureReason] == expected


def test_tag_constants():
    assert MAX_TAG_NAME_LENGTH == 50
    assert FLAGGED_TAG_NAME == "flagged"


@pytest.mark.parametrize(
    ("name", "source", "description"),
    [
        (QAFailureReason.NEVER_MOVED.value, "QA", "No movement detected"),
        ("needs-review", "USER", None),
        ("LARGE_GAP", "USER", None),
        ("never_moved", "USER", None),
    ],
)
def test_tag_count_takes_source_and_description_from_the_name(
    name, source, description
):
    other_source = "USER" if source == "QA" else "QA"
    count = RecordingTagCount.model_validate(
        {"name": name, "source": other_source, "recording_count": 3}
    )

    assert count.model_dump(mode="json") == {
        "name": name,
        "recording_count": 3,
        "source": source,
        "description": description,
    }


def test_tag_is_immutable():
    with pytest.raises(ValidationError):
        REVIEW_TAG.name = "other"


def test_assigning_notes_keeps_the_value():
    metadata = RecordingMetadata(tags=[REVIEW_TAG])

    metadata.notes = "checked by hand"

    assert metadata.notes == "checked by hand"
    assert metadata.tags == [REVIEW_TAG]


def test_lower_case_reason_name_is_a_separate_user_tag():
    user_tag = RecordingTag(name="never_moved", source=RecordingTagSource.USER)

    metadata = RecordingMetadata(tags=[user_tag, NEVER_MOVED_TAG])

    assert metadata.tags == [NEVER_MOVED_TAG, user_tag]
    assert metadata.tag_names == ["NEVER_MOVED", "never_moved"]
