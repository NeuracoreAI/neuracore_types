# Pending Release Notes

<!--
This file contains a human-written summary for the next release.
Append your changes below. This content will be included at the top of the release changelog.

Example: "This release adds support for multi-GPU training and improves streaming performance by 40%."
-->

## Summary

<!-- Append your summary here -->
`TraceStatusUpdates` has an optional `uploaded_files` list of `CompletedFileUpload`
(`upload_id`, `crc32c`), so clients can report the multipart uploads they finished.
