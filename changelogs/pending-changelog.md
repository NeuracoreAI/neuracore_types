# Pending Release Notes

<!--
This file contains a human-written summary for the next release.
Append your changes below. This content will be included at the top of the release changelog.

Example: "This release adds support for multi-GPU training and improves streaming performance by 40%."
-->

## Summary

<!-- Append your summary here -->

Timestamps are integer ticks of one microsecond on the recording time base, and
`TICKS_PER_SECOND` is exported by the Python and npm packages. A float
timestamp on input is read as legacy seconds and converted to ticks. Synchronized
episodes and recording documents carry the tick rate and the data clock window
in ticks, the recording start request carries the tick rate and start tick, and
the stop request carries the end tick. This is a breaking change: `timestamp`
values are ticks, not seconds.
