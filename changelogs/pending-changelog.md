# Pending Release Notes

<!--
This file contains a human-written summary for the next release.
Append your changes below. This content will be included at the top of the release changelog.

Example: "This release adds support for multi-GPU training and improves streaming performance by 40%."
-->

## Summary

<!-- Append your summary here -->

Unreleased TypeScript types are now published to npm under the `main` dist-tag on
every push to `main`, so consumers that need to track unreleased types can
`npm install @neuracore/types@main` instead of installing the repo from git.

Dataset statistics are now calculated asynchronously. `CalculateDatasetStatisticsRequest`,
`DatasetStatisticsJob` and `DatasetStatisticsJobStatus` describe the new job-based
flow, and `SynchronizedDatasetStatistics` is now a response-only model whose
`dataset_statistics` field is required.
