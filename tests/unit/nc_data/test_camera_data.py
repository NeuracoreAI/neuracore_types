"""Tests for CameraData and their batched variants."""

import numpy as np
import torch

from neuracore_types import (
    BatchedDepthData,
    BatchedRGBData,
    DepthCameraData,
    RGBCameraData,
)


class TestRGBCameraData:
    """Tests for RGBCameraData functionality."""

    def test_sample(self):
        """Test RGBCameraData.sample() creates valid instance."""
        data = RGBCameraData.sample()
        assert isinstance(data, RGBCameraData)
        assert isinstance(data.frame, np.ndarray)
        assert data.frame.shape == (480, 640, 3)
        assert data.frame.dtype == np.uint8
        assert data.intrinsics.shape == (3, 3)
        assert data.extrinsics.shape == (4, 4)

    def test_calculate_statistics(self):
        """Test calculate_statistics() for RGB camera data."""
        frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        data = RGBCameraData(
            frame=frame,
            intrinsics=np.ones((3, 3), dtype=np.float32),
            extrinsics=np.ones((4, 4), dtype=np.float32),
        )
        stats = data.calculate_statistics()

        assert stats.type == "CameraDataStats"
        assert stats.frame is not None
        assert stats.intrinsics.mean.shape == (3, 3)
        assert stats.extrinsics.mean.shape == (4, 4)

    def test_serialization(self):
        """Test JSON serialization of RGB data."""
        frame = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        data = RGBCameraData(
            frame=frame,
            intrinsics=np.ones((3, 3), dtype=np.float32),
            extrinsics=np.ones((4, 4), dtype=np.float32),
        )

        json_str = data.model_dump_json()
        loaded = RGBCameraData.model_validate_json(json_str)

        # Frame should be encoded/decoded correctly
        assert loaded.frame.shape == frame.shape
        assert np.all(loaded.intrinsics == 1.0)
        assert np.all(loaded.extrinsics == 1.0)

    def test_small_image(self):
        """Test RGB data with small image."""
        frame = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        data = RGBCameraData(
            frame=frame,
            intrinsics=np.ones((3, 3), dtype=np.float32),
            extrinsics=np.ones((4, 4), dtype=np.float32),
        )

        assert data.frame.shape == (10, 10, 3)

    def test_large_image(self):
        """Test RGB data with large image."""
        frame = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
        data = RGBCameraData(
            frame=frame,
            intrinsics=np.ones((3, 3), dtype=np.float32),
            extrinsics=np.ones((4, 4), dtype=np.float32),
        )

        assert data.frame.shape == (1080, 1920, 3)


class TestDepthCameraData:
    """Tests for DepthCameraData functionality."""

    def test_sample(self):
        """Test DepthCameraData.sample() creates valid instance."""
        data = DepthCameraData.sample()
        assert isinstance(data, DepthCameraData)
        assert isinstance(data.frame, np.ndarray)
        assert data.frame.shape == (480, 640)
        assert data.frame.dtype == np.float32

    def test_serialization(self):
        """Test JSON serialization of depth data."""
        frame = np.random.randn(50, 50).astype(np.float32)
        data = DepthCameraData(
            frame=frame,
            intrinsics=np.ones((3, 3), dtype=np.float32),
            extrinsics=np.ones((4, 4), dtype=np.float32),
        )

        json_str = data.model_dump_json()
        loaded = DepthCameraData.model_validate_json(json_str)

        assert loaded.frame.shape == frame.shape
        assert np.allclose(loaded.intrinsics, data.intrinsics)

    def test_depth_with_zeros(self):
        """Test depth data with zero values."""
        frame = np.zeros((100, 100), dtype=np.float32)
        data = DepthCameraData(
            frame=frame,
            intrinsics=np.ones((3, 3), dtype=np.float32),
            extrinsics=np.ones((4, 4), dtype=np.float32),
        )

        assert np.allclose(data.frame, 0.0)

    def test_depth_with_negative_values(self):
        """Test depth data with negative values."""
        frame = np.random.randn(100, 100).astype(np.float32)
        data = DepthCameraData(
            frame=frame,
            intrinsics=np.ones((3, 3), dtype=np.float32),
            extrinsics=np.ones((4, 4), dtype=np.float32),
        )

        assert data.frame.shape == (100, 100)


class TestBatchedRGBData:
    """Tests for BatchedRGBData functionality."""

    def test_from_nc_data(self):
        """Test BatchedRGBData.from_nc_data() conversion."""
        frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        rgb_data = RGBCameraData(
            frame=frame,
            intrinsics=np.ones((3, 3), dtype=np.float32),
            extrinsics=np.ones((4, 4), dtype=np.float32),
        )
        batched = BatchedRGBData.from_nc_data(rgb_data)

        assert isinstance(batched, BatchedRGBData)
        assert batched.frame.shape == (1, 1, 3, 100, 100)
        assert batched.intrinsics.shape == (1, 1, 3, 3)
        assert batched.extrinsics.shape == (1, 1, 4, 4)

    def test_sample(self):
        """Test BatchedRGBData.sample() with different dimensions."""
        batched = BatchedRGBData.sample(batch_size=2, time_steps=3)
        assert batched.frame.shape == (2, 3, 3, 224, 224)
        assert batched.intrinsics.shape == (2, 3, 3, 3)
        assert batched.extrinsics.shape == (2, 3, 4, 4)

    def test_sample_single_dimension(self):
        """Test sample with single batch and timestep."""
        batched = BatchedRGBData.sample(batch_size=1, time_steps=1)
        assert batched.frame.shape == (1, 1, 3, 224, 224)

    def test_to_device(self):
        """Test moving BatchedRGBData to different device."""
        batched = BatchedRGBData.sample(batch_size=1, time_steps=2)
        batched_cpu = batched.to(torch.device("cpu"))

        assert batched_cpu.frame.device.type == "cpu"
        assert batched_cpu.intrinsics.device.type == "cpu"
        assert batched_cpu.extrinsics.device.type == "cpu"

    def test_can_serialize_deserialize(self):
        """Test JSON serialization and deserialization."""
        batched = BatchedRGBData.sample(batch_size=2, time_steps=2)
        json_str = batched.model_dump_json()
        loaded = BatchedRGBData.model_validate_json(json_str)

        assert torch.equal(loaded.frame, batched.frame)
        assert loaded.frame.shape == batched.frame.shape


class TestBatchedDepthData:
    """Tests for BatchedDepthData functionality."""

    def test_from_nc_data(self):
        """Test BatchedDepthData.from_nc_data() conversion."""
        frame = np.random.randn(100, 100).astype(np.float32)
        depth_data = DepthCameraData(
            frame=frame,
            intrinsics=np.ones((3, 3), dtype=np.float32),
            extrinsics=np.ones((4, 4), dtype=np.float32),
        )
        batched = BatchedDepthData.from_nc_data(depth_data)

        assert isinstance(batched, BatchedDepthData)
        assert batched.frame.shape == (1, 1, 1, 100, 100)
        assert batched.intrinsics.shape == (1, 1, 3, 3)

    def test_sample(self):
        """Test BatchedDepthData.sample() with different dimensions."""
        batched = BatchedDepthData.sample(batch_size=3, time_steps=2)
        assert batched.frame.shape == (3, 2, 1, 224, 224)
        assert batched.intrinsics.shape == (3, 2, 3, 3)

    def test_to_device(self):
        """Test moving BatchedDepthData to different device."""
        batched = BatchedDepthData.sample(batch_size=2, time_steps=2)
        batched_cpu = batched.to(torch.device("cpu"))

        assert batched_cpu.frame.device.type == "cpu"
        assert batched_cpu.intrinsics.device.type == "cpu"

    def test_can_serialize_deserialize(self):
        """Test JSON serialization and deserialization."""
        batched = BatchedDepthData.sample(batch_size=2, time_steps=2)
        json_str = batched.model_dump_json()
        loaded = BatchedDepthData.model_validate_json(json_str)

        assert torch.equal(loaded.frame, batched.frame)
        assert loaded.frame.shape == batched.frame.shape
