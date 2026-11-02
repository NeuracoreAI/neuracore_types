# Pending Release Notes

<!--
This file contains a human-written summary for the next release.
Append your changes below. This content will be included at the top of the release changelog.

Example: "This release adds support for multi-GPU training and improves streaming performance by 40%."
-->

## Summary

<!-- Append your summary here -->
- Adds a `DONE` data type (`DoneData` / `BatchedDoneData`): a boolean episode done flag (0 while running, 1 once the task is complete) for training policies to predict when to stop.
