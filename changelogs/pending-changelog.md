# Pending Release Notes

<!--
This file contains a human-written summary for the next release.
Append your changes below. This content will be included at the top of the release changelog.

Example: "This release adds support for multi-GPU training and improves streaming performance by 40%."
-->

## Summary

<!-- Append your summary here -->

- Removes `GPUType.NVIDIA_H100_80GB`: H100 GPUs are no longer offered. Code that names it must use another GPU type, such as `GPUType.NVIDIA_A100_80GB`.
