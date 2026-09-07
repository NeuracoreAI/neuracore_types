# Pending Release Notes

<!--
This file contains a human-written summary for the next release.
Append your changes below. This content will be included at the top of the release changelog.

Example: "This release adds support for multi-GPU training and improves streaming performance by 40%."
-->

## Summary

<!-- Append your summary here -->

`TrainingCheckpoint` model describes a checkpoint a training job has available for download. `size_bytes` and `updated_at` are optional and may be unset.
Pending recordings now expose their cancellation state, defaulting to false for
existing recordings.
