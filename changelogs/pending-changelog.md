# Pending Release Notes

<!--
This file contains a human-written summary for the next release.
Append your changes below. This content will be included at the top of the release changelog.

Example: "This release adds support for multi-GPU training and improves streaming performance by 40%."
-->

## Summary

<!-- Append your summary here -->

Training jobs now carry a `deleted` flag and a `deleted_at` time. They mark a
job that is removed from the user view while the backend removes its cloud
resources in the background, so bulk deletion returns immediately and the job no
longer shows in training job lists.
