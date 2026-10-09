"""Tests for CameraData and their batched variants."""

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from neuracore_types import (
    BatchedDepthData,
    BatchedRGBData,
    DepthCameraData,
    RGBCameraData,
)
from neuracore_types.importer.config import (
    ImageChannelOrderConfig,
    ImageConventionConfig,
)
from neuracore_types.importer.data_config import (
    DataFormat,
    DepthCameraDataMappingItem,
    RGBCameraDataMappingItem,
)
from neuracore_types.importer.transform import DataTransformSequence
from neuracore_types.nc_data.camera_data import (
    DepthCameraDataImportConfig,
    RGBCameraDataImportConfig,
)


def _depth_frame(shape: tuple[int, int], seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 65536, shape, dtype=np.uint16)


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
        assert data.frame.dtype == np.uint16
        assert data.depth_scale_m == 1e-4

    def test_serialization_is_bit_exact_jpegxl(self):
        """Round trip uint16 depth through JPEG-XL JSON without changing a value."""
        frame = _depth_frame((50, 70))
        frame[:5, :5] = 0
        data = DepthCameraData(
            frame=frame,
            depth_scale_m=1e-4,
            intrinsics=np.ones((3, 3), dtype=np.float32),
            extrinsics=np.ones((4, 4), dtype=np.float32),
        )

        json_str = data.model_dump_json()
        loaded = DepthCameraData.model_validate_json(json_str)

        assert '"data:image/jxl;base64,' in json_str
        assert loaded.frame.dtype == np.uint16
        assert np.array_equal(loaded.frame, frame)
        assert loaded.depth_scale_m == 1e-4
        assert np.allclose(loaded.intrinsics, data.intrinsics)

    @pytest.mark.parametrize(
        "dtype", [np.float16, np.float32, np.float64, np.int32, np.uint8]
    )
    def test_rejects_other_dtypes(self, dtype):
        """Reject frames that are not uint16."""
        with pytest.raises(ValidationError):
            DepthCameraData(frame=np.ones((4, 4), dtype=dtype), depth_scale_m=1e-3)

    def test_statistics_are_in_metres(self):
        """Compute depth statistics on metres, not sensor units."""
        data = DepthCameraData(
            frame=np.full((2, 2), 2000, dtype=np.uint16), depth_scale_m=1e-3
        )

        stats = data.calculate_statistics()

        assert np.allclose(stats.frame.mean, 2.0)


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

    def test_transform_nc_data(self):
        """Test that transform_nc_data can be called without error."""
        frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        rgb_data = RGBCameraData(
            frame=frame,
            intrinsics=np.ones((3, 3), dtype=np.float32),
            extrinsics=np.ones((4, 4), dtype=np.float32),
        )
        batched = BatchedRGBData.from_nc_data(rgb_data)
        batched.transform_nc_data()
        # Check that frame is now of size (224, 224) after transformation
        assert batched.frame.shape == (1, 1, 3, 224, 224)

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
        assert batched.frame.dtype == torch.float32
        assert batched.intrinsics.dtype == torch.float32
        assert batched.extrinsics.dtype == torch.float32

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

    def test_from_nc_data_list_followed_by_transform(self):
        """Test that from_nc_data_list can be followed by transform_nc_data."""
        frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        rgb_data_list = [
            RGBCameraData(
                frame=frame,
                intrinsics=np.ones((3, 3), dtype=np.float32),
                extrinsics=np.ones((4, 4), dtype=np.float32),
            )
            for _ in range(5)
        ]
        batched = BatchedRGBData.from_nc_data_list(rgb_data_list)
        batched.transform_nc_data()

        # After transformation, frame should be resized to (224, 224)
        assert batched.frame.shape == (1, 5, 3, 224, 224)


class TestBatchedDepthData:
    """Tests for BatchedDepthData functionality."""

    def test_from_nc_data(self):
        """Build uint16 frames and the scale from one DepthCameraData."""
        frame = _depth_frame((100, 100))
        depth_data = DepthCameraData(
            frame=frame,
            depth_scale_m=1e-4,
            intrinsics=np.ones((3, 3), dtype=np.float32),
            extrinsics=np.ones((4, 4), dtype=np.float32),
        )
        batched = BatchedDepthData.from_nc_data(depth_data)

        assert isinstance(batched, BatchedDepthData)
        assert batched.frame.shape == (1, 1, 1, 100, 100)
        assert batched.frame.dtype == torch.uint16
        assert torch.equal(batched.frame[0, 0, 0], torch.from_numpy(frame))
        assert batched.depth_scale.shape == (1, 1)
        assert batched.depth_scale.dtype == torch.float32
        assert batched.intrinsics.shape == (1, 1, 3, 3)

    def test_from_nc_data_handles_none_extrinsics(self):
        """Test BatchedDepthData.from_nc_data() handles None extrinsics."""
        depth_data = DepthCameraData(
            frame=_depth_frame((100, 100)),
            depth_scale_m=1e-3,
            intrinsics=None,
            extrinsics=None,
        )
        batched = BatchedDepthData.from_nc_data(depth_data)
        assert batched.extrinsics.shape == (1, 1, 4, 4)
        assert batched.intrinsics.shape == (1, 1, 3, 3)

    def test_transform_nc_data_resizes_with_nearest_and_keeps_dtype(self):
        """Resize to 224 by 224 with nearest neighbour, so values and holes survive."""
        frame = np.array([[0, 500], [1000, 0]], dtype=np.uint16)
        batched = BatchedDepthData.from_nc_data(
            DepthCameraData(frame=frame, depth_scale_m=1e-3)
        )

        batched.transform_nc_data()

        assert batched.frame.shape == (1, 1, 1, 224, 224)
        assert batched.frame.dtype == torch.uint16
        assert set(batched.frame.flatten().tolist()) == {0, 500, 1000}

    def test_sample(self):
        """Test BatchedDepthData.sample() with different dimensions."""
        batched = BatchedDepthData.sample(batch_size=3, time_steps=2)
        assert batched.frame.shape == (3, 2, 1, 224, 224)
        assert batched.frame.dtype == torch.uint16
        assert batched.depth_scale.shape == (3, 2)
        assert batched.intrinsics.shape == (3, 2, 3, 3)

    def test_to_device(self):
        """Test moving BatchedDepthData to different device."""
        batched = BatchedDepthData.sample(batch_size=2, time_steps=2)
        batched_cpu = batched.to(torch.device("cpu"))

        assert batched_cpu.frame.device.type == "cpu"
        assert batched_cpu.depth_scale.device.type == "cpu"
        assert batched_cpu.intrinsics.device.type == "cpu"

    def test_can_serialize_deserialize(self):
        """Round trip uint16 frames and the scale through JSON."""
        batched = BatchedDepthData.from_nc_data_list([
            DepthCameraData(frame=_depth_frame((8, 9), seed=i), depth_scale_m=1e-3)
            for i in range(2)
        ])
        json_str = batched.model_dump_json()
        loaded = BatchedDepthData.model_validate_json(json_str)

        assert loaded.frame.dtype == torch.uint16
        assert torch.equal(loaded.frame, batched.frame)
        assert torch.equal(loaded.depth_scale, batched.depth_scale)

    def test_to_compute_dtype_gives_metres_and_is_idempotent(self):
        """Convert each frame to float32 metres with its own scale, once."""
        batched = BatchedDepthData.from_nc_data_list([
            DepthCameraData(
                frame=np.array([[0, 1000]], dtype=np.uint16), depth_scale_m=1e-3
            ),
            DepthCameraData(
                frame=np.array([[0, 1000]], dtype=np.uint16), depth_scale_m=1e-4
            ),
        ])

        batched.to_compute_dtype()
        batched.to_compute_dtype()

        assert batched.frame.dtype == torch.float32
        assert torch.allclose(batched.frame[0, 0, 0], torch.tensor([[0.0, 1.0]]))
        assert torch.allclose(batched.frame[0, 1, 0], torch.tensor([[0.0, 0.1]]))

    def test_batches_concatenate_frames_and_scales(self):
        """Concatenate uint16 frames and scales along the batch dimension."""
        first = BatchedDepthData.from_nc_data(
            DepthCameraData(frame=_depth_frame((4, 4)), depth_scale_m=1e-3)
        )
        second = BatchedDepthData.sample(batch_size=1, time_steps=1)
        second.frame = second.frame[..., :4, :4]

        batched = BatchedDepthData(
            frame=torch.cat([first.frame, second.frame], dim=0),
            depth_scale=torch.cat([first.depth_scale, second.depth_scale], dim=0),
            extrinsics=torch.cat([first.extrinsics, second.extrinsics], dim=0),
            intrinsics=torch.cat([first.intrinsics, second.intrinsics], dim=0),
        )
        batched.to_compute_dtype()

        assert batched.frame.shape == (2, 1, 1, 4, 4)
        assert torch.equal(batched.frame[1], torch.zeros(1, 1, 4, 4))
        assert torch.allclose(
            batched.frame[0, 0, 0], first.frame[0, 0, 0].to(torch.float32) * 1e-3
        )

    def test_from_nc_data_list_multiple_items(self):
        """Stack frames and scales along the time dimension."""
        depth_data_list = [
            DepthCameraData(
                frame=_depth_frame((100, 100), seed=i),
                depth_scale_m=1e-4 * (i + 1),
                intrinsics=np.ones((3, 3), dtype=np.float32),
                extrinsics=np.ones((4, 4), dtype=np.float32),
            )
            for i in range(10)
        ]
        batched = BatchedDepthData.from_nc_data_list(depth_data_list)

        assert batched.frame.shape == (1, 10, 1, 100, 100)
        assert batched.frame.dtype == torch.uint16
        assert batched.depth_scale.shape == (1, 10)
        assert torch.allclose(batched.depth_scale[0, 9], torch.tensor(1e-3))
        assert batched.intrinsics.shape == (1, 10, 3, 3)
        assert batched.extrinsics.shape == (1, 10, 4, 4)

    def test_from_nc_data_list_preserves_depth_values(self):
        """Keep exact uint16 values for every frame."""
        frames = [np.full((50, 50), v, dtype=np.uint16) for v in (1500, 2500)]
        batched = BatchedDepthData.from_nc_data_list(
            [DepthCameraData(frame=f, depth_scale_m=1e-3) for f in frames]
        )

        assert torch.equal(batched.frame[0, 0, 0], torch.from_numpy(frames[0]))
        assert torch.equal(batched.frame[0, 1, 0], torch.from_numpy(frames[1]))

    def test_from_nc_data_list_handles_none_extrinsics(self):
        """Test from_nc_data_list with None extrinsics/intrinsics."""
        depth_data = DepthCameraData(
            frame=_depth_frame((100, 100)),
            depth_scale_m=1e-3,
            intrinsics=None,
            extrinsics=None,
        )
        batched = BatchedDepthData.from_nc_data_list([depth_data])
        assert batched.extrinsics.shape == (1, 1, 4, 4)
        assert batched.intrinsics.shape == (1, 1, 3, 3)


class TestRGBCameraDataImportConfig:
    """Tests for RGBCameraDataImportConfig class."""

    def test_rgb_camera_data_import_config_defaults(self):
        """Test RGBCameraDataImportConfig with default format."""
        data_point = RGBCameraDataImportConfig(
            source="camera",
            mapping=[RGBCameraDataMappingItem(name="image")],
            format=DataFormat(
                image_convention=ImageConventionConfig.CHANNELS_LAST,
                order_of_channels=ImageChannelOrderConfig.RGB,
                normalized_pixel_values=False,
            ),
        )
        assert len(data_point.mapping) == 1
        frame = (
            np.stack([np.eye(100, 100, dtype=np.uint8) for _ in range(3)], axis=2) * 255
        )
        transforms = data_point.mapping[0].transforms
        transformed_data = transforms(frame)
        assert transformed_data.shape == (100, 100, 3)
        assert transformed_data.dtype == np.uint8
        assert transformed_data.min() == 0
        assert transformed_data.max() == 255

    def test_rgb_camera_data_import_config_normalized_pixel_values(self):
        """Test RGBCameraDataImportConfig transforms for channels_last RGB."""
        data_point = RGBCameraDataImportConfig(
            source="camera",
            mapping=[RGBCameraDataMappingItem(name="image")],
            format=DataFormat(normalized_pixel_values=True),
        )
        frame = np.stack([np.eye(100, 100, dtype=np.uint8) for _ in range(3)], axis=2)
        transforms = data_point.mapping[0].transforms
        transformed_data = transforms(frame)
        assert transformed_data.shape == (100, 100, 3)
        assert transformed_data.dtype == np.uint8
        assert transformed_data.min() == 0
        assert transformed_data.max() == 255

    def test_rgb_camera_data_import_config_channels_first(self):
        """Test RGBCameraDataImportConfig transforms for channels_first."""
        data_point = RGBCameraDataImportConfig(
            source="camera",
            mapping=[RGBCameraDataMappingItem(name="image")],
            format=DataFormat(image_convention=ImageConventionConfig.CHANNELS_FIRST),
        )
        frame = (
            np.stack([np.eye(100, 100, dtype=np.uint8) for _ in range(3)], axis=0) * 255
        )
        transforms = data_point.mapping[0].transforms
        transformed_data = transforms(frame)
        assert transformed_data.shape == (100, 100, 3)
        assert transformed_data.dtype == np.uint8
        assert transformed_data.min() == 0
        assert transformed_data.max() == 255

    def test_rgb_camera_data_import_config_transforms_bgr(self):
        """Test RGBCameraDataImportConfig transforms for BGR order."""
        data_point = RGBCameraDataImportConfig(
            source="camera",
            mapping=[RGBCameraDataMappingItem(name="image")],
            format=DataFormat(order_of_channels=ImageChannelOrderConfig.BGR),
        )
        frame_r = np.ones((100, 100), dtype=np.uint8) * 255
        frame_g = np.ones((100, 100), dtype=np.uint8) * 255
        frame_b = np.zeros((100, 100), dtype=np.uint8)
        frame = np.stack([frame_r, frame_g, frame_b], axis=2)
        transforms = data_point.mapping[0].transforms
        transformed_data = transforms(frame)
        assert transformed_data.shape == (100, 100, 3)
        assert transformed_data.dtype == np.uint8
        assert transformed_data.min() == 0
        assert transformed_data.max() == 255
        assert transformed_data[..., 0].max() == 0
        assert transformed_data[..., 1].max() == 255
        assert transformed_data[..., 2].max() == 255


class TestDepthCameraDataImportConfig:
    """Tests for DepthCameraDataImportConfig class."""

    @staticmethod
    def _transforms(**format_fields: object) -> DataTransformSequence:
        data_point = DepthCameraDataImportConfig(
            source="depth",
            mapping=[DepthCameraDataMappingItem(name="depth_image")],
            format=DataFormat(**format_fields),
        )
        return data_point.mapping[0].transforms

    def test_integer_source_keeps_values(self):
        """Keep D405 tenth of a millimetre units as uint16, 65535 included."""
        frame = np.array([[0, 5000], [65535, 1]], dtype=np.int32)

        result = self._transforms(depth_scale_m=0.0001)(frame)

        assert result.dtype == np.uint16
        assert np.array_equal(result, frame)

    def test_missing_scale_raises(self):
        """Refuse a depth config without depth_scale_m."""
        with pytest.raises(ValueError, match="depth_scale_m"):
            self._transforms()

    def test_squeezes_singleton_channel(self):
        """Accept HxWx1 and 1xHxW images."""
        transforms = self._transforms(depth_scale_m=0.001)

        assert transforms(np.ones((10, 12, 1), dtype=np.uint16)).shape == (10, 12)
        assert transforms(np.ones((1, 10, 12), dtype=np.uint16)).shape == (10, 12)
