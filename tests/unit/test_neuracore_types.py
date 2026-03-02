"""Tests for data ordering functionality.

This module contains tests to verify that the data ordering utilities
work correctly and ensure consistent ordering across sync points.
"""

import time

import numpy as np
import pytest

from neuracore_types import (
    Custom1DData,
    DataType,
    DepthCameraData,
    JointData,
    LanguageData,
    PointCloudData,
    PoseData,
    RGBCameraData,
    SynchronizedEpisode,
    SynchronizedPoint,
)

SYNCHRONIZED_POINT_UNORDERED = SynchronizedPoint(
    timestamp=time.time(),
    data={
        DataType.JOINT_POSITIONS: {
            "joint_2": JointData(value=0.1),
            "joint_1": JointData(value=0.2),
            "joint_3": JointData(value=0.3),
        },
        DataType.JOINT_VELOCITIES: {
            "joint_2": JointData(value=0.1),
            "joint_1": JointData(value=0.2),
            "joint_3": JointData(value=0.3),
        },
        DataType.POSES: {
            "pose_2": PoseData(pose=[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]),
            "pose_1": PoseData(pose=[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]),
        },
        DataType.RGB_IMAGES: {
            "camera_3": RGBCameraData(frame=np.zeros((224, 224, 3))),
            "camera_1": RGBCameraData(frame=np.zeros((224, 224, 3))),
            "camera_2": RGBCameraData(frame=np.zeros((224, 224, 3))),
        },
        DataType.DEPTH_IMAGES: {
            "depth_2": DepthCameraData(frame=np.zeros((224, 224))),
            "depth_1": DepthCameraData(frame=np.zeros((224, 224))),
        },
        DataType.POINT_CLOUDS: {
            "lidar_2": PointCloudData(
                points=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float16)
            ),
            "lidar_1": PointCloudData(points=np.array([[7, 8, 9]], dtype=np.float16)),
        },
        DataType.CUSTOM_1D: {
            "sensor_z": Custom1DData(data=np.array([1, 2, 3])),
            "sensor_a": Custom1DData(data=np.array([4, 5, 6])),
        },
        DataType.LANGUAGE: {"default": LanguageData(text="Pick up the object")},
    },
)

ORDER_SCHEMA = {
    DataType.JOINT_POSITIONS: {2: "joint_3", 0: "joint_1", 1: "joint_2"},
    DataType.JOINT_VELOCITIES: {1: "joint_2", 2: "joint_3", 0: "joint_1"},
    DataType.POSES: {1: "pose_2", 0: "pose_1"},
    DataType.RGB_IMAGES: {2: "camera_3", 0: "camera_1", 1: "camera_2"},
    DataType.DEPTH_IMAGES: {1: "depth_2", 0: "depth_1"},
    DataType.POINT_CLOUDS: {1: "lidar_2", 0: "lidar_1"},
    DataType.LANGUAGE: {0: "default"},
    DataType.CUSTOM_1D: {1: "sensor_z", 0: "sensor_a"},
}


def _verify_sync_point_ordering(sync_point_unordered: SynchronizedPoint):
    """Test that sync point ordering works correctly."""
    ordered_sync = sync_point_unordered.order(ORDER_SCHEMA)

    # Verify that all keys are now sorted
    assert DataType.JOINT_POSITIONS in ordered_sync.data
    assert list(ordered_sync[DataType.JOINT_POSITIONS].keys()) == [
        "joint_1",
        "joint_2",
        "joint_3",
    ]

    # Verify joint velocities ordering
    assert DataType.JOINT_VELOCITIES in ordered_sync.data
    assert list(ordered_sync[DataType.JOINT_VELOCITIES].keys()) == [
        "joint_1",
        "joint_2",
        "joint_3",
    ]

    # Verify poses ordering (both the dict keys and pose names should be sorted)
    assert DataType.POSES in ordered_sync.data
    assert list(ordered_sync[DataType.POSES].keys()) == ["pose_1", "pose_2"]

    # Verify RGB images ordering
    assert DataType.RGB_IMAGES in ordered_sync.data
    assert list(ordered_sync[DataType.RGB_IMAGES].keys()) == [
        "camera_1",
        "camera_2",
        "camera_3",
    ]

    # Verify depth images ordering
    assert DataType.DEPTH_IMAGES in ordered_sync.data
    assert list(ordered_sync[DataType.DEPTH_IMAGES].keys()) == ["depth_1", "depth_2"]

    # Verify point clouds ordering
    assert DataType.POINT_CLOUDS in ordered_sync.data
    assert list(ordered_sync[DataType.POINT_CLOUDS].keys()) == ["lidar_1", "lidar_2"]

    # Verify custom data ordering
    assert DataType.CUSTOM_1D in ordered_sync.data
    assert list(ordered_sync[DataType.CUSTOM_1D].keys()) == ["sensor_a", "sensor_z"]


def test_sync_point_ordering():
    """Test that sync point ordering works correctly."""
    _verify_sync_point_ordering(SYNCHRONIZED_POINT_UNORDERED)


def test_synced_data_ordering():
    """Test that synced data ordering works for multiple sync points."""
    # Create multiple unordered sync points
    sync_points = [SYNCHRONIZED_POINT_UNORDERED for _ in range(3)]

    # Modify timestamps to be different
    for i, sync_point in enumerate(sync_points):
        sync_point.timestamp = time.time() + i

    # Create synced data
    synced_data = SynchronizedEpisode(
        observations=sync_points,
        start_time=sync_points[0].timestamp,
        end_time=sync_points[-1].timestamp,
        robot_id="robot1",
    )

    ordered_synced_data = synced_data.order(ORDER_SCHEMA)

    for frame in ordered_synced_data.observations:
        _verify_sync_point_ordering(frame)


def test_sync_point_ordering_errors_on_extra_data_type():
    """Ordering should reject sync points with data types outside the schema."""
    sync_point = SynchronizedPoint.model_validate(
        SYNCHRONIZED_POINT_UNORDERED.model_dump()
    )
    sync_point.data[DataType.JOINT_TORQUES] = {"joint_1": JointData(value=0.4)}

    with pytest.raises(
        ValueError,
        match="SynchronizedPoint data types must exactly match embodiment_description",
    ):
        sync_point.order(ORDER_SCHEMA)


def test_sync_point_ordering_errors_on_missing_data_type():
    """Ordering should reject schemas that require a missing data type."""
    sync_point = SynchronizedPoint.model_validate(
        SYNCHRONIZED_POINT_UNORDERED.model_dump()
    )
    sync_point.data.pop(DataType.CUSTOM_1D)

    with pytest.raises(
        ValueError,
        match="SynchronizedPoint data types must exactly match embodiment_description",
    ):
        sync_point.order(ORDER_SCHEMA)


def test_sync_point_ordering_errors_on_extra_sensor_name():
    """Ordering should reject extra sensor names within a data type."""
    sync_point = SynchronizedPoint.model_validate(
        SYNCHRONIZED_POINT_UNORDERED.model_dump()
    )
    sync_point.data[DataType.JOINT_POSITIONS]["joint_extra"] = JointData(value=0.4)

    with pytest.raises(
        ValueError,
        match=(
            "SynchronizedPoint names for DataType .* "
            "must exactly match embodiment_description"
        ),
    ):
        sync_point.order(ORDER_SCHEMA)


def test_sync_point_ordering_errors_on_missing_sensor_name():
    """Ordering should reject missing sensor names within a data type."""
    sync_point = SynchronizedPoint.model_validate(
        SYNCHRONIZED_POINT_UNORDERED.model_dump()
    )
    sync_point.data[DataType.JOINT_POSITIONS].pop("joint_2")

    with pytest.raises(
        ValueError,
        match=(
            "SynchronizedPoint names for DataType .* "
            "must exactly match embodiment_description"
        ),
    ):
        sync_point.order(ORDER_SCHEMA)
