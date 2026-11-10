# Pending Release Notes

<!--
This file contains a human-written summary for the next release.
Append your changes below. This content will be included at the top of the release changelog.

Example: "This release adds support for multi-GPU training and improves streaming performance by 40%."
-->

## Summary

<!-- Append your summary here -->

Per-stream signalling epochs, all optional: `AvailableRobotInstance` and
`AvailableRobotCapacityUpdate` have `stream_epochs` (stream ID to epoch). Each
stream of a robot instance has its own registration epoch.
`AvailableRobotInstance.epoch` and `AvailableRobotCapacityUpdate.epochs` are now
the highest epoch of the instance's streams.
