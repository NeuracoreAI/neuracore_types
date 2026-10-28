# Pending Release Notes

<!--
This file contains a human-written summary for the next release.
Append your changes below. This content will be included at the top of the release changelog.

Example: "This release adds support for multi-GPU training and improves streaming performance by 40%."
-->

## Summary

<!-- Append your summary here -->

- `to_safe_name` now URL-encodes sensor names the way ROS escapes topic names (`/` becomes `%2F` and `.` becomes `%2E`) instead of escaping `/` as `\`. `from_safe_name` reads both formats, since older SDK versions still upload with `\`.
