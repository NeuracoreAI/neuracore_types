"""Unit tests for training VM disk size estimation."""

from neuracore_types import DataType, estimate_min_disk_size_gb
from neuracore_types.training.disk_size import DISK_SIZE_ROUND_GB, MIN_DISK_SIZE_GB


def _gib_to_bytes(gib: float) -> int:
    return int(gib * (1024**3))


def test_estimate_floors_at_100_gb_for_tiny_dataset():
    total = estimate_min_disk_size_gb(
        size_bytes=1024,
        num_demonstrations=2,
        hydra_arg_name="cnnmlp",
    )
    assert total == MIN_DISK_SIZE_GB


def test_estimate_video_cache_higher_than_non_video():
    size_bytes = _gib_to_bytes(40)
    video = estimate_min_disk_size_gb(
        size_bytes=size_bytes,
        num_demonstrations=20,
        hydra_arg_name="cnnmlp",
        input_cross_embodiment_description={"robot": {DataType.RGB_IMAGES: {0: "cam"}}},
    )
    non_video = estimate_min_disk_size_gb(
        size_bytes=size_bytes,
        num_demonstrations=20,
        hydra_arg_name="cnnmlp",
        input_cross_embodiment_description={
            "robot": {DataType.JOINT_POSITIONS: {0: "joint"}}
        },
    )
    assert video > non_video


def test_estimate_steady_cache_uses_max_multiplier():
    from neuracore_types.training.disk_size import (
        POINT_CLOUD_CACHE_MULTIPLIER,
        VIDEO_CACHE_MULTIPLIER,
        _steady_cache_gb,
    )

    size_gib = 10.0
    video_and_pc = _steady_cache_gb(
        size_gib, [DataType.RGB_IMAGES, DataType.POINT_CLOUDS]
    )
    video_only = _steady_cache_gb(size_gib, [DataType.RGB_IMAGES])
    pc_only = _steady_cache_gb(size_gib, [DataType.POINT_CLOUDS])

    assert video_and_pc == size_gib * VIDEO_CACHE_MULTIPLIER
    assert video_and_pc == video_only
    assert pc_only == size_gib * POINT_CLOUD_CACHE_MULTIPLIER


def test_estimate_rounds_up_to_50_gb_increments():
    total = estimate_min_disk_size_gb(
        size_bytes=_gib_to_bytes(200),
        num_demonstrations=50,
        hydra_arg_name="pi05",
        input_cross_embodiment_description={"robot": {DataType.RGB_IMAGES: {0: "cam"}}},
    )
    assert total >= MIN_DISK_SIZE_GB
    assert total % DISK_SIZE_ROUND_GB == 0
