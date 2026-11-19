"""Tests for TraceStatusUpdates."""

from neuracore_types import (
    CompletedFileUpload,
    RecordingDataTraceStatus,
    TraceStatusUpdates,
)


def test_uploaded_files_are_parsed_but_not_written_to_the_trace() -> None:
    updates = TraceStatusUpdates.model_validate({
        "status": "UPLOAD_COMPLETE",
        "uploaded_bytes": 1024,
        "total_bytes": 1024,
        "uploaded_files": [
            {"upload_id": "upload-1", "crc32c": "mnG7TA=="},
            {"upload_id": "upload-2", "crc32c": "4waSgw=="},
        ],
    })

    assert updates.uploaded_files == [
        CompletedFileUpload(upload_id="upload-1", crc32c="mnG7TA=="),
        CompletedFileUpload(upload_id="upload-2", crc32c="4waSgw=="),
    ]
    changes = updates.minimal_changes()
    assert changes.keys() == {"status", "uploaded_at", "uploaded_bytes", "total_bytes"}
    assert changes["status"] == RecordingDataTraceStatus.UPLOAD_COMPLETE.value
