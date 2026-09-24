# Pending Release Notes

<!--
This file contains a human-written summary for the next release.
Append your changes below. This content will be included at the top of the release changelog.

Example: "This release adds support for multi-GPU training and improves streaming performance by 40%."
-->

## Summary

<!-- Append your summary here -->

This release replaces QA pass/fail result unions with QAFinding (reason, optional interval, optional trace), adds stillness and tracking-error failure reasons, and adds a human-readable `description` to QAFinding.

Adds `Codec.H264_FAST` (`h264_fast`) to the recorded-video codec enum, alongside the existing `h264_medium` and `h264_lossless` options.
