"""Tests for CameraData and their batched variants."""

import numpy as np
import torch

from neuracore_types import (
    BatchedDepthData,
    BatchedRGBData,
    DepthCameraData,
    RGBCameraData,
)
from neuracore_types.importer.config import (
    DistanceUnitsConfig,
    ImageChannelOrderConfig,
    ImageConventionConfig,
)
from neuracore_types.importer.data_config import DataFormat, MappingItem
from neuracore_types.importer.transform import (
    CastToNumpyDtype,
    Clip,
    ImageChannelOrder,
    ImageFormat,
    NanToNum,
    Scale,
    Unnormalize,
)
from neuracore_types.nc_data.camera_data import (
    DepthCameraDataImportConfig,
    RGBCameraDataImportConfig,
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

    def test_from_nc_data_list_single_item(self):
        """Test from_nc_data_list with single RGB image."""
        frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        rgb_data = RGBCameraData(
            frame=frame,
            intrinsics=np.ones((3, 3), dtype=np.float32),
            extrinsics=np.ones((4, 4), dtype=np.float32),
        )
        batched = BatchedRGBData.from_nc_data_list([rgb_data])

        assert isinstance(batched, BatchedRGBData)
        assert batched.frame.shape == (1, 1, 3, 100, 100)
        assert batched.intrinsics.shape == (1, 1, 3, 3)
        assert batched.extrinsics.shape == (1, 1, 4, 4)

    def test_from_nc_data_list_multiple_items(self):
        """Test from_nc_data_list with multiple RGB images."""
        frames = [
            np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8) for _ in range(5)
        ]
        rgb_data_list = [
            RGBCameraData(
                frame=frame,
                intrinsics=np.ones((3, 3), dtype=np.float32) * (i + 1),
                extrinsics=np.ones((4, 4), dtype=np.float32) * (i + 1),
            )
            for i, frame in enumerate(frames)
        ]
        batched = BatchedRGBData.from_nc_data_list(rgb_data_list)

        assert batched.frame.shape == (1, 5, 3, 100, 100)
        assert batched.intrinsics.shape == (1, 5, 3, 3)
        assert batched.extrinsics.shape == (1, 5, 4, 4)
        # Check that intrinsics were stacked correctly
        assert torch.allclose(batched.intrinsics[0, 0], torch.ones(3, 3) * 1.0)
        assert torch.allclose(batched.intrinsics[0, 4], torch.ones(3, 3) * 5.0)

    def test_from_nc_data_list_large_batch(self):
        """Test from_nc_data_list with large number of images."""
        num_images = 100
        rgb_data_list = [RGBCameraData.sample() for _ in range(num_images)]
        batched = BatchedRGBData.from_nc_data_list(rgb_data_list)

        assert batched.frame.shape[0] == 1  # Batch dimension
        assert batched.frame.shape[1] == num_images  # Time dimension
        assert batched.frame.shape[2] == 3  # Channels

    def test_from_nc_data_list_preserves_image_content(self):
        """Test that from_nc_data_list preserves exact image content."""
        # Create images with distinct patterns
        frame1 = np.zeros((50, 50, 3), dtype=np.uint8)
        frame1[:, :, 0] = 255  # Red channel
        frame2 = np.zeros((50, 50, 3), dtype=np.uint8)
        frame2[:, :, 1] = 255  # Green channel

        rgb_data_list = [
            RGBCameraData(
                frame=frame1,
                intrinsics=np.eye(3, dtype=np.float32),
                extrinsics=np.eye(4, dtype=np.float32),
            ),
            RGBCameraData(
                frame=frame2,
                intrinsics=np.eye(3, dtype=np.float32),
                extrinsics=np.eye(4, dtype=np.float32),
            ),
        ]
        batched = BatchedRGBData.from_nc_data_list(rgb_data_list)

        # Check red channel of first image
        assert torch.all(batched.frame[0, 0, 0] == 255)
        assert torch.all(batched.frame[0, 0, 1] == 0)
        # Check green channel of second image
        assert torch.all(batched.frame[0, 1, 0] == 0)
        assert torch.all(batched.frame[0, 1, 1] == 255)

    def test_from_nc_data_list_handles_none_extrinsics(self):
        """Test from_nc_data_list with None extrinsics/intrinsics."""
        rgb_data = RGBCameraData(
            frame=np.zeros((50, 50, 3), dtype=np.uint8),
            intrinsics=None,
            extrinsics=None,
        )
        batched = BatchedRGBData.from_nc_data_list([rgb_data])

        # Should create zero tensors for None values
        assert batched.intrinsics.shape == (1, 1, 3, 3)
        assert batched.extrinsics.shape == (1, 1, 4, 4)


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

    def test_from_nc_data_list_single_item(self):
        """Test from_nc_data_list with single depth image."""
        frame = np.random.randn(100, 100).astype(np.float32)
        depth_data = DepthCameraData(
            frame=frame,
            intrinsics=np.ones((3, 3), dtype=np.float32),
            extrinsics=np.ones((4, 4), dtype=np.float32),
        )
        batched = BatchedDepthData.from_nc_data_list([depth_data])

        assert isinstance(batched, BatchedDepthData)
        assert batched.frame.shape == (1, 1, 1, 100, 100)
        assert batched.intrinsics.shape == (1, 1, 3, 3)
        assert batched.extrinsics.shape == (1, 1, 4, 4)

    def test_from_nc_data_list_multiple_items(self):
        """Test from_nc_data_list with multiple depth images."""
        frames = [np.random.randn(100, 100).astype(np.float32) for _ in range(10)]
        depth_data_list = [
            DepthCameraData(
                frame=frame,
                intrinsics=np.ones((3, 3), dtype=np.float32),
                extrinsics=np.ones((4, 4), dtype=np.float32),
            )
            for frame in frames
        ]
        batched = BatchedDepthData.from_nc_data_list(depth_data_list)

        assert batched.frame.shape == (1, 10, 1, 100, 100)
        assert batched.intrinsics.shape == (1, 10, 3, 3)
        assert batched.extrinsics.shape == (1, 10, 4, 4)

    def test_from_nc_data_list_preserves_depth_values(self):
        """Test that from_nc_data_list preserves exact depth values."""
        frame1 = np.ones((50, 50), dtype=np.float32) * 1.5
        frame2 = np.ones((50, 50), dtype=np.float32) * 2.5

        depth_data_list = [
            DepthCameraData(
                frame=frame1,
                intrinsics=np.eye(3, dtype=np.float32),
                extrinsics=np.eye(4, dtype=np.float32),
            ),
            DepthCameraData(
                frame=frame2,
                intrinsics=np.eye(3, dtype=np.float32),
                extrinsics=np.eye(4, dtype=np.float32),
            ),
        ]
        batched = BatchedDepthData.from_nc_data_list(depth_data_list)

        assert torch.allclose(batched.frame[0, 0, 0], torch.ones(50, 50) * 1.5)
        assert torch.allclose(batched.frame[0, 1, 0], torch.ones(50, 50) * 2.5)


class TestRGBCameraDataImportConfig:
    """Tests for RGBCameraDataImportConfig class."""

    def test_rgb_camera_data_import_config_defaults(self):
        """Test RGBCameraDataImportConfig with default format."""
        data_point = RGBCameraDataImportConfig(source="camera")
        assert data_point.source == "camera"
        assert len(data_point.mapping) == 0

    def test_rgb_camera_data_import_config_transforms_channels_last_rgb(self):
        """Test RGBCameraDataImportConfig transforms for channels_last RGB."""
        data_point = RGBCameraDataImportConfig(
            source="camera",
            mapping=[MappingItem(name="image")],
            format=DataFormat(
                image_convention=ImageConventionConfig.CHANNELS_LAST,
                order_of_channels=ImageChannelOrderConfig.RGB,
                normalized_pixel_values=False,
            ),
        )
        assert len(data_point.mapping) == 1
        transforms = data_point.mapping[0].transforms.transforms
        assert isinstance(transforms[0], Clip)
        assert isinstance(transforms[1], CastToNumpyDtype)

    def test_rgb_camera_data_import_config_transforms_channels_first(self):
        """Test RGBCameraDataImportConfig transforms for channels_first."""
        data_point = RGBCameraDataImportConfig(
            source="camera",
            mapping=[MappingItem(name="image")],
            format=DataFormat(image_convention=ImageConventionConfig.CHANNELS_FIRST),
        )
        transforms = data_point.mapping[0].transforms.transforms
        # ImageFormat is added after Clip, CastToNumpyDtype
        assert isinstance(transforms[2], ImageFormat)

    def test_rgb_camera_data_import_config_transforms_bgr(self):
        """Test RGBCameraDataImportConfig transforms for BGR order."""
        data_point = RGBCameraDataImportConfig(
            source="camera",
            mapping=[MappingItem(name="image")],
            format=DataFormat(order_of_channels=ImageChannelOrderConfig.BGR),
        )
        transforms = data_point.mapping[0].transforms.transforms
        assert isinstance(transforms[-1], ImageChannelOrder)

    def test_rgb_camera_data_import_config_transforms_normalized(self):
        """Test RGBCameraDataImportConfig transforms for normalized pixel values."""
        data_point = RGBCameraDataImportConfig(
            source="camera",
            mapping=[MappingItem(name="image")],
            format=DataFormat(normalized_pixel_values=True),
        )
        transforms = data_point.mapping[0].transforms.transforms
        assert isinstance(transforms[0], Unnormalize)
        assert isinstance(transforms[1], Clip)


class TestDepthCameraDataImportConfig:
    """Tests for DepthCameraDataImportConfig class."""

    def test_depth_camera_data_import_config_meters(self):
        """Test DepthCameraDataImportConfig with meters."""
        data_point = DepthCameraDataImportConfig(
            source="depth",
            mapping=[MappingItem(name="depth_image")],
            format=DataFormat(distance_units=DistanceUnitsConfig.M),
        )
        transforms = data_point.mapping[0].transforms.transforms
        assert isinstance(transforms[0], NanToNum)
        assert not any(isinstance(t, Scale) for t in transforms)

    def test_depth_camera_data_import_config_millimeters(self):
        """Test DepthCameraDataImportConfig with millimeters."""
        data_point = DepthCameraDataImportConfig(
            source="depth",
            mapping=[MappingItem(name="depth_image")],
            format=DataFormat(distance_units=DistanceUnitsConfig.MM),
        )
        transforms = data_point.mapping[0].transforms.transforms
        assert any(isinstance(t, Scale) for t in transforms)
