# Pending Release Notes

<!--
This file contains a human-written summary for the next release.
Append your changes below. This content will be included at the top of the release changelog.

Example: "This release adds support for multi-GPU training and improves streaming performance by 40%."
-->

## Summary

Finer training job lifecycle statuses (queued → provisioning → starting → syncing → fetching → calculating statistics → batch-size tuning → training), with phase progress on the job. Legacy PREPARING_DATA / PENDING / RUNNING are kept for in-flight jobs and are not rewritten.
