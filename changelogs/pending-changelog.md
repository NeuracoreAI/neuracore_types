# Pending Release Notes

<!--
This file contains a human-written summary for the next release.
Append your changes below. This content will be included at the top of the release changelog.

Example: "This release adds support for multi-GPU training and improves streaming performance by 40%."
-->

## Summary

<!-- Append your summary here -->
- `RecordingMetadata.tags` holds user tags and one QA tag per QA failure reason.
- Adds `RecordingTagCount` with the QA reason `description`, `RecordingTagRequest`, `is_qa_tag_name` and the `tag_count` sort field to list, add, remove and sort recordings by tag.
- `QAFailureReason` members are declared in display order, most important first. Tags are stored with QA tags first in that order, then user tags in the order they were added.
- Removes `RecordingStatus` and the `status` and `qa_flag_reasons` fields of `RecordingMetadata`. Use `RecordingMetadata.tags`, which is now required, together with `tag_names` and `tag_count`. Input that still holds `status` or `qa_flag_reasons` is accepted and these keys are ignored. This is a breaking change.
