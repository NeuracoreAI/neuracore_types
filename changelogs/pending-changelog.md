# Pending Release Notes

<!--
This file contains a human-written summary for the next release.
Append your changes below. This content will be included at the top of the release changelog.

Example: "This release adds support for multi-GPU training and improves streaming performance by 40%."
-->

## Summary

<!-- Append your summary here -->

Samples carry `timestamp_us`, an integer count of microseconds on the recording
clock. `timestamp` stays in seconds and is derived from `timestamp_us`.
Synchronized episodes carry `start_timestamp_us` and `end_timestamp_us`.
Recordings, the recording start request and the recording stop request carry
the recording window in microseconds. `MICROSECONDS_PER_SECOND` is exported by
the Python and npm packages. The package needs pydantic 2.12 or later.
