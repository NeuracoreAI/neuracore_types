# Pending Release Notes

<!--
This file contains a human-written summary for the next release.
Append your changes below. This content will be included at the top of the release changelog.

Example: "This release adds support for multi-GPU training and improves streaming performance by 40%."
-->

## Summary

<!-- Append your summary here -->

- Adds `PolicyRollout`, `PolicyRolloutSeries` and `PolicyRolloutsResponse` for the policy playback console. `PolicyRollout.images` holds a download URL for each camera frame the policy saw. `PolicyRolloutSeries` rejects NaN and infinite values, and `predicted` and `ground_truth` of different lengths.
- Adds `PolicyRolloutEpoch` for each epoch that saves validation rollouts. `PolicyRolloutEpoch.status` tells whether its rollouts are `UPLOADING`, `READY` or `FAILED`.
- Adds `SignallingTokenRequest` and `SignallingTokenResponse` for the signalling session token endpoint.
- Training jobs now carry `root_job_id` (the first job of their family of runs) and `resumed_from_epoch` (the epoch of the checkpoint they continue from), so resumed runs can be grouped and show their own progress.
- `Recording.trace_json_size_bytes` records the size of each sensor's trace.json (not its videos) at finalize, empty for recordings saved before it existed.
