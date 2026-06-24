"""Tests for PointCloudData, wire format, and BatchedPointCloudData."""

import json
from typing import cast

import numpy as np
import pytest
import torch

from neuracore_types import BatchedPointCloudData, DataType, PointCloudData
from neuracore_types.importer.config import DistanceUnitsConfig
from neuracore_types.importer.data_config import DataFormat, PointCloudDataMappingItem
from neuracore_types.importer.transform import Scale
from neuracore_types.nc_data import DATA_TYPE_CONTENT_MAPPING
from neuracore_types.nc_data.point_cloud_data import (
    PointCloudDataImportConfig,
    decode_point_cloud_frame,
    decode_point_cloud_wire_metadata,
    encode_point_cloud_frame_parts,
)


def encode_wire_frame(data: PointCloudData) -> bytes:
    header, metadata_json, points_view, rgb_view = encode_point_cloud_frame_parts(data)
    parts: list[bytes | memoryview] = [header, metadata_json, points_view]
    if rgb_view is not None:
        parts.append(rgb_view)
    return b"".join(parts)


class TestPointCloudData:
    """Tests for PointCloudData functionality."""

    def test_sample(self):
        """Test PointCloudData.sample() creates valid instance."""
        data = PointCloudData.sample()
        assert isinstance(data, PointCloudData)
        assert isinstance(data.points, np.ndarray)
        assert data.points.shape == (1000, 3)
        assert data.points.dtype == np.float16
        assert data.type == "PointCloudData"

    def test_calculate_statistics(self):
        """Test calculate_statistics() for point cloud."""
        points = np.random.randn(50, 3).astype(np.float16)
        data = PointCloudData(points=points)
        stats = data.calculate_statistics()

        assert stats.type == "PointCloudDataStats"
        assert stats.points is not None
        assert len(stats.points.mean) == 3
        assert len(stats.points.std) == 3

    def test_serialization(self):
        """Test JSON serialization of point cloud data."""
        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float16)
        data = PointCloudData(points=points)

        json_str = data.model_dump_json()
        loaded = PointCloudData.model_validate_json(json_str)

        assert np.allclose(loaded.points, data.points)

    def test_base64_encoding_decoding(self):
        """Test base64 encoding/decoding of point cloud."""
        points = np.random.randn(10, 3).astype(np.float16)
        data = PointCloudData(points=points)

        # Serialize to dict with base64
        data_dict = json.loads(data.model_dump_json())
        assert isinstance(data_dict["points"], str)

        # Deserialize from dict
        loaded = PointCloudData.model_validate(data_dict)
        assert np.allclose(loaded.points, points)

    def test_empty_points(self):
        """Test handling of empty point cloud."""
        with pytest.raises(Exception):
            PointCloudData(points=np.array([], dtype=np.float16).reshape(0, 3))

    def test_invalid_shape(self):
        """Test that invalid point cloud shape is handled."""
        with pytest.raises(Exception):
            # Should fail if not Nx3
            PointCloudData(points=np.array([[1.0, 2.0]], dtype=np.float16))

    def test_large_points(self):
        """Test handling of large point clouds."""
        large_pc = np.random.randn(10000, 3).astype(np.float16)
        data = PointCloudData(points=large_pc)

        assert data.points.shape == (10000, 3)
        stats = data.calculate_statistics()
        assert stats.points is not None

    def test_single_point(self):
        """Test point cloud with single point."""
        single_point = np.array([[1.0, 2.0, 3.0]], dtype=np.float16)
        data = PointCloudData(points=single_point)

        assert data.points.shape == (1, 3)


class TestPointCloudWire:
    """Tests for binary wire encode/decode."""

    def test_roundtrip_xyz_only(self):
        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float16)
        original = PointCloudData(timestamp=1.5, points=points)
        decoded = decode_point_cloud_frame(encode_wire_frame(original))
        np.testing.assert_array_equal(decoded.points, original.points)
        assert decoded.timestamp == original.timestamp
        assert decoded.rgb_points is None

    def test_roundtrip_xyz_rgb_calib(self):
        n = 100
        points = np.random.randn(n, 3).astype(np.float16)
        rgb = np.random.randint(0, 256, (n, 3), dtype=np.uint8)
        extrinsics = np.eye(4, dtype=np.float16)
        intrinsics = np.eye(3, dtype=np.float16)
        original = PointCloudData(
            timestamp=2.0,
            points=points,
            rgb_points=rgb,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
        )
        decoded = decode_point_cloud_frame(encode_wire_frame(original))
        np.testing.assert_array_equal(decoded.points, original.points)
        np.testing.assert_array_equal(decoded.rgb_points, original.rgb_points)
        np.testing.assert_array_equal(decoded.extrinsics, original.extrinsics)
        np.testing.assert_array_equal(decoded.intrinsics, original.intrinsics)

    def test_max_size_307200_cloud(self):
        n = 640 * 480
        points = np.zeros((n, 3), dtype=np.float16)
        rgb = np.zeros((n, 3), dtype=np.uint8)
        original = PointCloudData(timestamp=3.0, points=points, rgb_points=rgb)
        wire = encode_wire_frame(original)
        decoded = decode_point_cloud_frame(wire)
        assert decoded.points.shape == (n, 3)
        assert decoded.rgb_points is not None
        assert decoded.rgb_points.shape == (n, 3)

    def test_decode_invalid_json_metadata_rejected(self):
        payload = json.dumps({"type": "PointCloudData", "points": "abc"}).encode()
        with pytest.raises(ValueError, match="metadata length"):
            decode_point_cloud_frame(payload)

    def test_decode_truncated_payload(self):
        wire = encode_wire_frame(
            PointCloudData(
                points=np.zeros((10, 3), dtype=np.float16),
                timestamp=1.0,
            )
        )
        with pytest.raises(ValueError):
            decode_point_cloud_frame(wire[:-10])

    def test_encode_requires_points(self):
        with pytest.raises(ValueError, match="points are required"):
            encode_point_cloud_frame_parts(PointCloudData(timestamp=1.0, points=None))

    def test_content_mapping_marks_point_cloud_binary(self):
        assert DATA_TYPE_CONTENT_MAPPING[DataType.POINT_CLOUDS] == "POINT_CLOUD"
        assert DATA_TYPE_CONTENT_MAPPING[DataType.JOINT_POSITIONS] == "JSON"

    def test_decode_point_cloud_wire_metadata(self):
        points = np.array([[1.0, 2.0, 3.0]], dtype=np.float16)
        wire = encode_wire_frame(PointCloudData(timestamp=1.5, points=points))
        metadata, metadata_end = decode_point_cloud_wire_metadata(wire)
        assert metadata["timestamp"] == 1.5
        assert metadata["num_points"] == 1
        assert metadata_end > 4

    def test_trace_json_metadata_validates_like_camera_trace(self):
        trace_json = [
            {
                "type": "PointCloudData",
                "timestamp": 1.0,
                "frame_idx": 0,
                "offset": 0,
                "length": 10,
            },
            {
                "type": "PointCloudData",
                "timestamp": 2.0,
                "frame_idx": 1,
                "offset": 10,
                "length": 12,
            },
        ]
        frames = [PointCloudData.model_validate(item) for item in trace_json]
        assert len(frames) == 2
        assert frames[0].frame_idx == 0
        assert frames[0].points is None
        assert frames[1].frame_idx == 1


class TestBatchedPointCloudData:
    """Tests for BatchedPointCloudData functionality."""

    def test_from_nc_data(self):
        """Test BatchedPointCloudData.from_nc_data() conversion."""
        points = np.random.randn(50, 3).astype(np.float16)
        pc_data = PointCloudData(points=points)
        batched = BatchedPointCloudData.from_nc_data(pc_data)

        assert isinstance(batched, BatchedPointCloudData)
        assert batched.points.shape == (1, 1, 3, 50)
        assert np.allclose(batched.points[0, 0].permute(-1, -2).numpy(), points)

    def test_sample(self):
        """Test BatchedPointCloudData.sample() with different dimensions."""
        batched = BatchedPointCloudData.sample(batch_size=2, time_steps=3)
        assert batched.points.shape == (2, 3, 3, 1000)

    def test_sample_single_batch(self):
        """Test sample with single batch and timestep."""
        batched = BatchedPointCloudData.sample(batch_size=1, time_steps=1)
        assert batched.points.shape == (1, 1, 3, 1000)

    def test_to_device(self):
        """Test moving to different device."""
        batched = BatchedPointCloudData.sample(batch_size=2, time_steps=2)
        batched_cpu = batched.to(torch.device("cpu"))
        batched_cpu = cast(BatchedPointCloudData, batched_cpu)

        assert batched_cpu.points.device.type == "cpu"
        assert torch.allclose(batched_cpu.points, batched.points)

    def test_from_nc_data_preserves_values(self):
        """Test that from_nc_data preserves exact values."""
        points = np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], dtype=np.float16)
        pc_data = PointCloudData(points=points)
        batched = BatchedPointCloudData.from_nc_data(pc_data)
        batched = cast(BatchedPointCloudData, batched)
        assert np.allclose(
            batched.points[0, 0].permute(-1, -2).numpy(), points, rtol=1e-5
        )

    def test_can_serialize_deserialize(self):
        """Test JSON serialization and deserialization of BatchedPointCloudData."""
        batched = BatchedPointCloudData.sample(batch_size=2, time_steps=2)
        json_str = batched.model_dump_json()
        loaded = BatchedPointCloudData.model_validate_json(json_str)

        assert torch.equal(loaded.points, batched.points)
        assert loaded.points.shape == batched.points.shape


class TestPointCloudDataStatistics:
    """Tests for PointCloudData statistics."""

    def test_statistics_shape(self):
        """Test that statistics have correct shape."""
        points = np.random.randn(100, 3).astype(np.float16)
        data = PointCloudData(points=points)
        stats = data.calculate_statistics()

        assert len(stats.points.mean) == 3
        assert len(stats.points.std) == 3
        assert len(stats.points.min) == 3
        assert len(stats.points.max) == 3

    def test_statistics_values(self):
        """Test that statistics contain sensible values."""
        points = np.random.randn(1000, 3).astype(np.float16)
        data = PointCloudData(points=points)
        stats = data.calculate_statistics()

        # Mean should be close to 0 for random normal data
        assert np.allclose(stats.points.mean, 0.0, atol=0.2)
        # Std should be close to 1 for random normal data
        assert np.allclose(stats.points.std, 1.0, atol=0.2)

    def test_statistics_min_max(self):
        """Test that min and max are correctly calculated."""
        points = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [0.0, 1.0, 2.0],
            ],
            dtype=np.float16,
        )
        data = PointCloudData(points=points)
        stats = data.calculate_statistics()

        assert np.allclose(stats.points.min, [0.0, 1.0, 2.0])
        assert np.allclose(stats.points.max, [4.0, 5.0, 6.0])

    def test_statistics_concatenation(self):
        """Test that point cloud statistics can be concatenated."""
        points1 = np.random.randn(50, 3).astype(np.float16)
        points2 = np.random.randn(50, 3).astype(np.float16)

        data1 = PointCloudData(points=points1)
        data2 = PointCloudData(points=points2)

        stats1 = data1.calculate_statistics()
        stats2 = data2.calculate_statistics()

        concatenated = stats1.points.concatenate(stats2.points)
        assert len(concatenated.mean) == 6  # 3 + 3


class TestPointCloudDataImportConfig:
    """Tests for PointCloudDataImportConfig class."""

    def test_point_cloud_data_import_config_meters(self):
        """Test PointCloudDataImportConfig with meters."""
        data_point = PointCloudDataImportConfig(
            source="point_cloud",
            mapping=[PointCloudDataMappingItem(name="points")],
            format=DataFormat(distance_units=DistanceUnitsConfig.M),
        )
        transforms = data_point.mapping[0].transforms.transforms
        assert not any(isinstance(t, Scale) for t in transforms)

    def test_point_cloud_data_import_config_millimeters(self):
        """Test PointCloudDataImportConfig with millimeters."""
        data_point = PointCloudDataImportConfig(
            source="point_cloud",
            mapping=[PointCloudDataMappingItem(name="points")],
            format=DataFormat(distance_units=DistanceUnitsConfig.MM),
        )
        transforms = data_point.mapping[0].transforms.transforms
        assert any(isinstance(t, Scale) for t in transforms)
