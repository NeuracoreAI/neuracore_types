# Pending Release Notes

<!--
This file contains a human-written summary for the next release.
Append your changes below. This content will be included at the top of the release changelog.

Example: "This release adds support for multi-GPU training and improves streaming performance by 40%."
-->

## Summary

<!-- Append your summary here -->

Import sessions: `ImportSession`, `ImportSessionCreateRequest`,
`ImportSessionItemState` and `ImportSessionItemKeysPage`. A session records how
many recordings an import selected (`expected_recording_count`) and reports
`recording_count` and `is_complete` from the item keys saved so far.

Recording provenance: `RecordingSource` is a discriminated union of
`RobotInstanceSource` (`instance`) and `ImportSessionSource`
(`import_session_id`, `item_key`), tagged by `type` (`RecordingSourceType`).
`RecordingStartRequest` now takes `source` in place of `instance`; a request
with a top-level `instance`, as older clients send, still records that robot
instance, and `RecordingStartRequest.instance` still reads it (None for an
import). `Recording` and `PendingRecording` have an optional `source`, None
for recordings started before this release. `Recording.instance` is now
`int | None` and defaults to None: it mirrors a robot instance source and is
None for imported recordings, which leave it out of their JSON so older clients
read their default instead of failing validation.

Recording notifications: `BaseRecodingUpdatePayload.instance` is now optional
and None for imported recordings, so SAVED, DISCARDED and EXPIRED can report
them. START and STOP payloads still require an instance, since only live robot
recordings start and stop.
