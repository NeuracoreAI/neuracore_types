# Pending Release Notes

<!--
This file contains a human-written summary for the next release.
Append your changes below. This content will be included at the top of the release changelog.

Example: "This release adds support for multi-GPU training and improves streaming performance by 40%."
-->

## Summary

<!-- Append your summary here -->

Recording STOP notifications carry the recording's `end_time` in a new
`RecordingStopPayload`.

Finer training job lifecycle statuses (queued → provisioning → starting → syncing → fetching → calculating statistics → batch-size tuning → training), with phase progress on the job. Legacy PREPARING_DATA / PENDING / RUNNING are kept for in-flight jobs and are not rewritten.

`TrainingProgress` / `TrainingPhaseProgress` models group epoch, step, status, and phase progress (`num_completed_items` / `num_total_items`) for training job updates.
