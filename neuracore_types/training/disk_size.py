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

# Built .pt samples under ~/.neuracore/training/sample_cache (dataset_sample_cache).
# Additive to the recording/PNG cache: warm hits skip decode but do not evict
# frames. Heuristic vs compressed dataset size; resized float tensors can still
# exceed mp4 footprint across every training timestep.
SAMPLE_CACHE_MULTIPLIER = 1.5
SAMPLE_CACHE_POINT_CLOUD_MULTIPLIER = 2.0  # Dense point-cloud tensors vs source

_VIDEO_DATA_TYPES = frozenset({DataType.RGB_IMAGES, DataType.DEPTH_IMAGES})
_POINT_CLOUD_DATA_TYPES = frozenset({DataType.POINT_CLOUDS})

# Training configs, need to update if training config.yaml in SDK changes.
# Prefetch queue bound is 2 * max_prefetch_decode_workers (default 4 → 8).
MAX_PREFETCH_WORKERS = 8  # Cap on concurrent staged videos awaiting/in decode
KEEP_LAST_N = 2  # Epoch checkpoints kept (matches keep_last_n_checkpoints)

# Pretrained size and per-checkpoint size. Local peak during save =
# single_checkpoint_gb * (1 + KEEP_LAST_N)  # new write before old delete
# Pretrained = max(_DEFAULT_PRETRAINED_GB, archive_gib * 1.5) from GCS archives.
#
# TODO: Prefer exact sizes from algorithm metadata / Firestore when available
# (especially custom algorithms); hydra_arg_name tiers are a coarse fallback.
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
    descriptions: list[CrossEmbodimentDescription],
) -> set[DataType]:
    """Union DataType keys from one or more cross-embodiment descriptions.

    Args:
        descriptions: Input and/or output cross-embodiment maps. Empty
            descriptions contribute nothing. Keys may be ``DataType`` enums or
            their string values.

    Returns:
        The set of distinct data types present across all embodiments.
    """
    data_types: set[DataType] = set()
    for description in descriptions:
        for embodiment in description.values():
            for key in embodiment:
                data_types.add(DataType(key) if not isinstance(key, DataType) else key)
    return data_types


def _algorithm_disk_tiers(hydra_arg_name: str | None) -> tuple[float, float]:
    """Look up pretrained and per-checkpoint disk tiers for an algorithm.

    Args:
        hydra_arg_name: Algorithm hydra config name (e.g. ``"pi05"``,
            ``"cnnmlp"``). Case-insensitive. Unknown or missing names fall back
            to the default tiers.

    Returns:
        ``(pretrained_gb, single_checkpoint_gb)`` reserved for weights and one
        checkpoint file respectively.
    """
    if not hydra_arg_name:
        return _DEFAULT_PRETRAINED_GB, _DEFAULT_CHECKPOINT_GB
    return _ALGORITHM_DISK_TIERS.get(
        hydra_arg_name.lower(),
        (_DEFAULT_PRETRAINED_GB, _DEFAULT_CHECKPOINT_GB),
    )


def _steady_cache_gb(size_gib: float, data_types: Iterable[DataType]) -> float:
    """Estimate the steady-state recording/frame cache footprint.

    Covers decoded frames under ``recording_cache`` (PNG for RGB/depth, tensors
    for point clouds). Uses the max applicable modality multiplier so stacked
    modalities do not over-count the shared recording bytes.

    Args:
        size_gib: Dataset size in GiB (typically compressed source bytes).
        data_types: Modalities selected for training inputs/outputs.

    Returns:
        Estimated GiB for the persistent frame cache after prefetch completes.
    """
    data_type_set = set(data_types)
    multipliers = [NON_VIDEO_CACHE_MULTIPLIER]
    if data_type_set & _VIDEO_DATA_TYPES:
        multipliers.append(VIDEO_CACHE_MULTIPLIER)
    if data_type_set & _POINT_CLOUD_DATA_TYPES:
        multipliers.append(POINT_CLOUD_CACHE_MULTIPLIER)
    return size_gib * max(multipliers)


def _sample_cache_gb(size_gib: float, data_types: Iterable[DataType]) -> float:
    """Estimate disk for fully built training samples (torch.save ``.pt`` entries).

    Additive to the recording/PNG cache: warm hits skip decode but do not evict
    frames. Enabled by default via ``dataset_sample_cache``. Train/val usually
    share one cache when worker-side preprocessing matches (e.g. both
    ResizePad only); the safety buffer covers divergent worker-side configs.

    Args:
        size_gib: Dataset size in GiB (typically compressed source bytes).
        data_types: Modalities selected for training inputs/outputs.

    Returns:
        Estimated GiB for ``~/.neuracore/training/sample_cache``, or ``0.0``
        when ``size_gib`` is non-positive.
    """
    if size_gib <= 0:
        return 0.0
    multiplier = SAMPLE_CACHE_MULTIPLIER
    if set(data_types) & _POINT_CLOUD_DATA_TYPES:
        multiplier = max(multiplier, SAMPLE_CACHE_POINT_CLOUD_MULTIPLIER)
    return size_gib * multiplier


def _video_prefetch_staging_gb(
    size_gib: float,
    num_demonstrations: int,
    data_types: Iterable[DataType],
) -> float:
    """Estimate temporary disk for concurrent video download/decode staging.

    Only applies when RGB/depth are selected. Models concurrent staging as
    ``min(MAX_PREFETCH_WORKERS, N)`` average-sized recordings (aligned with the
    prefetch decode queue bound of ``2 * max_prefetch_decode_workers``).

    Args:
        size_gib: Dataset size in GiB (typically compressed source bytes).
        num_demonstrations: Number of recordings/demonstrations in the dataset.
        data_types: Modalities selected for training inputs/outputs.

    Returns:
        Estimated GiB of transient mp4/staging space during prefetch, or
        ``0.0`` when video is unused or inputs are empty.
    """
    if not (set(data_types) & _VIDEO_DATA_TYPES):
        return 0.0
    n = max(num_demonstrations, 0)
    if n <= 0 or size_gib <= 0:
        return 0.0
    workers = min(MAX_PREFETCH_WORKERS, n)
    return workers * (size_gib / n)


def _round_disk_size_gb(raw_gb: float) -> int:
    """Ceil, round up to ``DISK_SIZE_ROUND_GB``, and floor at ``MIN_DISK_SIZE_GB``.

    Args:
        raw_gb: Unrounded estimate including the safety buffer.

    Returns:
        Recommended disk size in whole GB.
    """
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

    Formula::

        total = OS_BASE + steady_cache + sample_cache + prefetch_staging
              + pretrained + checkpoints + artifacts + safety_buffer
        then ceil, round up to 50 GB, floor at 100 GB.

    Args:
        size_bytes: Total dataset size in bytes.
        num_demonstrations: Number of recordings/demonstrations.
        hydra_arg_name: Optional algorithm name for pretrained/checkpoint tiers.
        input_cross_embodiment_description: Selected input sensors per robot.
        output_cross_embodiment_description: Selected output sensors per robot.

    Returns:
        Minimum recommended boot-disk size in GB.
    """
    size_gib = max(size_bytes, 0) / (1024**3)
    descriptions = [
        description
        for description in (
            input_cross_embodiment_description,
            output_cross_embodiment_description,
        )
        if description
    ]
    data_types = _collect_data_types(descriptions)
    pretrained_gb, single_checkpoint_gb = _algorithm_disk_tiers(hydra_arg_name)
    checkpoints_gb = single_checkpoint_gb * (1 + KEEP_LAST_N)
    artifacts_gb = ARTIFACTS_GB + single_checkpoint_gb

    steady_cache_gb = _steady_cache_gb(size_gib, data_types)
    sample_cache_gb = _sample_cache_gb(size_gib, data_types)
    prefetch_staging_gb = _video_prefetch_staging_gb(
        size_gib, num_demonstrations, data_types
    )

    subtotal = (
        OS_BASE_GB
        + steady_cache_gb
        + sample_cache_gb
        + prefetch_staging_gb
        + pretrained_gb
        + checkpoints_gb
        + artifacts_gb
    )
    return _round_disk_size_gb(subtotal + SAFETY_BUFFER_RATIO * subtotal)
