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
