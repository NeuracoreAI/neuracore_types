# Pending Release Notes

<!--
This file contains a human-written summary for the next release.
Append your changes below. This content will be included at the top of the release changelog.

Example: "This release adds support for multi-GPU training and improves streaming performance by 40%."
-->

## Summary

<!-- Append your summary here -->

Regional signalling fields, all optional: `SignallingTokenRequest` takes
`robot_instance` and `register_epoch`, `SignallingTokenResponse` returns
`region_endpoints` (for an organization token), `AvailableRobotInstance` has
`epoch`, and `AvailableRobotCapacityUpdate` has `epochs` (robot ID to instance
to epoch). An epoch is a time in microseconds since the Unix epoch.

- `RecordingMetadata.tags` holds user tags and one QA tag per QA failure reason, `RecordingMetadata.tag_names` lists their names, and the legacy `status` and `qa_flag_reasons` stay in step with the tags.
- Adds `RecordingTagCount` with the QA reason `description`, `RecordingTagRequest` and `is_qa_tag_name` to list, add and remove recording tags.
- `QAFailureReason` members are declared in display order, most important first. Tags are stored with QA tags first in that order, then user tags in the order they were added.
