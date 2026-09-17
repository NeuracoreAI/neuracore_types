"""Estimate minimum training VM disk size from dataset and algorithm factors."""

from __future__ import annotations

import math
from collections.abc import Iterable

from neuracore_types.episode.episode import CrossEmbodimentDescription
from neuracore_types.nc_data import DataType

OS_BASE_GB = 50.0  # OS, packages, and non-training system overhead
ARTIFACTS_GB = 5.0  # Logs, metrics, exports (plus one checkpoint for model weights)
SAFETY_BUFFER_RATIO = 0.20  # Extra headroom as a fraction of the subtotal
MIN_DISK_SIZE_GB = 100  # Never provision a training VM smaller than this
DISK_SIZE_ROUND_GB = 50  # Round the final estimate up to this GB step

VIDEO_CACHE_MULTIPLIER = 1.25  # mp4 → PNG/cache expansion for RGB/depth
NON_VIDEO_CACHE_MULTIPLIER = 1.1  # Mild expansion for non-video modalities
POINT_CLOUD_CACHE_MULTIPLIER = 1.1  # Cache expansion when point clouds are selected

VIDEO_DATA_TYPES = frozenset({DataType.RGB_IMAGES, DataType.DEPTH_IMAGES})
POINT_CLOUD_DATA_TYPES = frozenset({DataType.POINT_CLOUDS})

# Training configs, need to update if training config.yaml in SDK changes.
MAX_PREFETCH_WORKERS = 8  # Cap on concurrent recording prefetch workers
KEEP_LAST_N = 5  # Epoch checkpoints kept in addition to checkpoint_latest

# Pretrained size and per-checkpoint size. Local peak =
# single_checkpoint_gb * (1 + KEEP_LAST_N)  # latest + keep_last_n
# Pretrained = max(_DEFAULT_PRETRAINED_GB, archive_gib * 1.5) from GCS archives.
_DEFAULT_PRETRAINED_GB = 10.0  # Minimum reserved space for pretrained weights
_DEFAULT_CHECKPOINT_GB = 10.0  # Unknown/custom single checkpoint
_ALGORITHM_DISK_TIERS: dict[str, tuple[float, float]] = {
    # (pretrained_gb, single_checkpoint_gb)
    "pi0": (10.0, 20.0),
    "pi05": (27.0, 25.0),
    "groot": (8.0, 15.0),
    "cnnmlp": (5.0, 2.0),
    "act": (5.0, 2.0),
    "diffusion_policy": (5.0, 3.0),
}


def _collect_data_types(
    *descriptions: CrossEmbodimentDescription | None,
) -> set[DataType]:
    data_types: set[DataType] = set()
    for description in descriptions:
        if not description:
            continue
        for embodiment in description.values():
            for key in embodiment:
                data_types.add(DataType(key) if not isinstance(key, DataType) else key)
    return data_types


def _algorithm_disk_tiers(hydra_arg_name: str | None) -> tuple[float, float]:
    """Return (pretrained_gb, single_checkpoint_gb) for an algorithm."""
    if not hydra_arg_name:
        return _DEFAULT_PRETRAINED_GB, _DEFAULT_CHECKPOINT_GB
    return _ALGORITHM_DISK_TIERS.get(
        hydra_arg_name.lower(),
        (_DEFAULT_PRETRAINED_GB, _DEFAULT_CHECKPOINT_GB),
    )


def _steady_cache_gb(size_gib: float, data_types: Iterable[DataType]) -> float:
    data_type_set = set(data_types)
    multipliers = [NON_VIDEO_CACHE_MULTIPLIER]
    if data_type_set & VIDEO_DATA_TYPES:
        multipliers.append(VIDEO_CACHE_MULTIPLIER)
    if data_type_set & POINT_CLOUD_DATA_TYPES:
        multipliers.append(POINT_CLOUD_CACHE_MULTIPLIER)
    return size_gib * max(multipliers)


def _prefetch_temp_gb(
    size_gib: float,
    num_demonstrations: int,
    data_types: Iterable[DataType],
) -> float:
    if not (set(data_types) & VIDEO_DATA_TYPES):
        return 0.0
    n = max(num_demonstrations, 0)
    if n <= 0 or size_gib <= 0:
        return 0.0
    workers = min(MAX_PREFETCH_WORKERS, n)
    return workers * (size_gib / n)


def _round_disk_size_gb(raw_gb: float) -> int:
    ceiled = math.ceil(raw_gb)
    rounded = math.ceil(ceiled / DISK_SIZE_ROUND_GB) * DISK_SIZE_ROUND_GB
    return max(MIN_DISK_SIZE_GB, rounded)


def estimate_min_disk_size_gb(
    size_bytes: int,
    num_demonstrations: int,
    hydra_arg_name: str | None = None,
    input_cross_embodiment_description: CrossEmbodimentDescription | None = None,
    output_cross_embodiment_description: CrossEmbodimentDescription | None = None,
) -> int:
    """Estimate the minimum training VM disk size in GB.

    Formula:
        total = OS_BASE + steady_cache + prefetch_temp + pretrained
              + checkpoints + artifacts + safety_buffer
        then ceil, round up to 50 GB, floor at 100 GB.
    """
    size_gib = max(size_bytes, 0) / (1024**3)
    data_types = _collect_data_types(
        input_cross_embodiment_description,
        output_cross_embodiment_description,
    )
    pretrained_gb, single_checkpoint_gb = _algorithm_disk_tiers(hydra_arg_name)
    checkpoints_gb = single_checkpoint_gb * (1 + KEEP_LAST_N)
    artifacts_gb = ARTIFACTS_GB + single_checkpoint_gb

    steady_cache_gb = _steady_cache_gb(size_gib, data_types)
    prefetch_temp_gb = _prefetch_temp_gb(size_gib, num_demonstrations, data_types)

    subtotal = (
        OS_BASE_GB
        + steady_cache_gb
        + prefetch_temp_gb
        + pretrained_gb
        + checkpoints_gb
        + artifacts_gb
    )
    return _round_disk_size_gb(subtotal + SAFETY_BUFFER_RATIO * subtotal)
