"""Unit tests for training VM disk size estimation."""

from neuracore_types import DataType, estimate_min_disk_size_gb
from neuracore_types.training.disk_size import (
    DISK_SIZE_ROUND_GB,
    MIN_DISK_SIZE_GB,
    POINT_CLOUD_CACHE_MULTIPLIER,
    SAMPLE_CACHE_MULTIPLIER,
    SAMPLE_CACHE_POINT_CLOUD_MULTIPLIER,
    VIDEO_CACHE_MULTIPLIER,
    _sample_cache_gb,
    _steady_cache_gb,
)


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
    # Large enough that video steady+prefetch outruns joint-only after 50 GB rounding.
    size_bytes = _gib_to_bytes(120)
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
    size_gib = 10.0
    video_and_pc = _steady_cache_gb(
        size_gib, [DataType.RGB_IMAGES, DataType.POINT_CLOUDS]
    )
    video_only = _steady_cache_gb(size_gib, [DataType.RGB_IMAGES])
    pc_only = _steady_cache_gb(size_gib, [DataType.POINT_CLOUDS])

    assert video_and_pc == size_gib * VIDEO_CACHE_MULTIPLIER
    assert video_and_pc == video_only
    assert pc_only == size_gib * POINT_CLOUD_CACHE_MULTIPLIER


def test_estimate_sample_cache_additive_and_point_cloud_aware():
    size_gib = 10.0
    joints = _sample_cache_gb(size_gib, [DataType.JOINT_POSITIONS])
    video = _sample_cache_gb(size_gib, [DataType.RGB_IMAGES])
    point_cloud = _sample_cache_gb(size_gib, [DataType.POINT_CLOUDS])

    assert joints == size_gib * SAMPLE_CACHE_MULTIPLIER
    assert video == size_gib * SAMPLE_CACHE_MULTIPLIER
    assert point_cloud == size_gib * SAMPLE_CACHE_POINT_CLOUD_MULTIPLIER
    assert _sample_cache_gb(0.0, [DataType.RGB_IMAGES]) == 0.0


def test_estimate_includes_sample_cache_in_total(monkeypatch):
    """Sample cache raises the estimate vs recording-cache-only baseline."""
    import sys

    disk_size_mod = sys.modules["neuracore_types.training.disk_size"]
    size_bytes = _gib_to_bytes(80)
    kwargs = dict(
        size_bytes=size_bytes,
        num_demonstrations=40,
        hydra_arg_name="cnnmlp",
        input_cross_embodiment_description={"robot": {DataType.RGB_IMAGES: {0: "cam"}}},
    )
    with_sample = estimate_min_disk_size_gb(**kwargs)
    monkeypatch.setattr(disk_size_mod, "_sample_cache_gb", lambda *args, **kwargs: 0.0)
    without_sample = estimate_min_disk_size_gb(**kwargs)
    assert with_sample > without_sample


def test_estimate_rounds_up_to_50_gb_increments():
    total = estimate_min_disk_size_gb(
        size_bytes=_gib_to_bytes(200),
        num_demonstrations=50,
        hydra_arg_name="pi05",
        input_cross_embodiment_description={"robot": {DataType.RGB_IMAGES: {0: "cam"}}},
    )
    assert total >= MIN_DISK_SIZE_GB
    assert total % DISK_SIZE_ROUND_GB == 0
