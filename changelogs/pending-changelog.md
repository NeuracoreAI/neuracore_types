# Pending Release Notes

<!--
This file contains a human-written summary for the next release.
Append your changes below. This content will be included at the top of the release changelog.

Example: "This release adds support for multi-GPU training and improves streaming performance by 40%."
-->

## Summary

<!-- Append your summary here -->

- Removes `RecordingStatus` and the `status` and `qa_flag_reasons` fields of `RecordingMetadata`. Use `RecordingMetadata.tags`, which is now required, together with `tag_names`. Input that still holds `status` or `qa_flag_reasons` is accepted and these keys are ignored. Also removes `tags_from_legacy` and `legacy_from_tags`. This is a breaking change.
- `RecordingMetadata` now validates each field assignment, so assigning an invalid value raises a `ValidationError`.
