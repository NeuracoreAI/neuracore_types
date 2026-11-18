# Pending Release Notes

<!--
This file contains a human-written summary for the next release.
Append your changes below. This content will be included at the top of the release changelog.

Example: "This release adds support for multi-GPU training and improves streaming performance by 40%."
-->

## Summary

<!-- Append your summary here -->

Import sessions: `ImportSession`, `ImportSessionStatus`,
`ImportSessionCreateRequest` and `ImportSessionCloseRequest`, plus
`IMPORT_SESSION_RESERVED_FLOOR` and `IMPORT_SESSION_STRIDE`, which describe the
robot instance range an import session's workers record from. `Recording` and
`PendingRecording` have optional `source` (`RecordingSource`) and
`import_session_id`; both are None for recordings started before this release.
