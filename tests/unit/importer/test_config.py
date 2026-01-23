"""Unit tests for config.py module."""

import json

import pytest
import yaml

from neuracore_types.importer.config import (
    AngleConfig,
    DistanceUnitsConfig,
    EulerOrderConfig,
    ImageChannelOrderConfig,
    ImageConventionConfig,
    IndexRangeConfig,
    LanguageConfig,
    NormalizeConfig,
    OrientationConfig,
    OutputDatasetConfig,
    PoseConfig,
    QuaternionOrderConfig,
    RobotConfig,
    RotationConfig,
    TorqueUnitsConfig,
)
from neuracore_types.importer.data_config import DataFormat, MappingItem
from neuracore_types.importer.transform import (
    CastToNumpyDtype,
    Clip,
    ImageFormat,
    NumpyToScalar,
)
from neuracore_types.nc_data import DatasetImportConfig, DataType
from neuracore_types.nc_data.camera_data import RGBCameraDataImportConfig
from neuracore_types.nc_data.joint_data import JointPositionsDataImportConfig


class TestOutputDatasetConfig:
    """Tests for OutputDatasetConfig class."""

    def test_output_dataset_config_basic(self):
        """Test basic OutputDatasetConfig creation."""
        dataset = OutputDatasetConfig(
            name="test_dataset", description="Test description"
        )
        assert dataset.name == "test_dataset"
        assert dataset.description == "Test description"
        assert dataset.tags == []

    def test_output_dataset_config_with_tags(self):
        """Test OutputDatasetConfig with tags."""
        dataset = OutputDatasetConfig(
            name="test_dataset", tags=["tag1", "tag2"], description="Test"
        )
        assert dataset.tags == ["tag1", "tag2"]


class TestRobotConfig:
    """Tests for RobotConfig class."""

    def test_robot_config_basic(self):
        """Test basic RobotConfig creation."""
        robot = RobotConfig(name="test_robot")
        assert robot.name == "test_robot"
        assert robot.urdf_path is None
        assert robot.mjcf_path is None
        assert robot.overwrite_existing is False

    def test_robot_config_with_paths(self):
        """Test RobotConfig with URDF and MJCF paths."""
        robot = RobotConfig(
            name="test_robot",
            urdf_path="/path/to/urdf",
            mjcf_path="/path/to/mjcf",
            overwrite_existing=True,
        )
        assert robot.urdf_path == "/path/to/urdf"
        assert robot.mjcf_path == "/path/to/mjcf"
        assert robot.overwrite_existing is True


class TestIndexRangeConfig:
    """Tests for IndexRangeConfig class."""

    def test_index_range_valid(self):
        """Test valid IndexRangeConfig."""
        index_range = IndexRangeConfig(start=0, end=10)
        assert index_range.start == 0
        assert index_range.end == 10

    def test_index_range_invalid(self):
        """Test invalid IndexRangeConfig where start > end."""
        with pytest.raises(ValueError, match="Index range start must be less than end"):
            IndexRangeConfig(start=10, end=0)


class TestNormalizeConfig:
    """Tests for NormalizeConfig class."""

    def test_normalize_config_defaults(self):
        """Test NormalizeConfig with default values."""
        config = NormalizeConfig()
        assert config.min == 0.0
        assert config.max == 1.0

    def test_normalize_config_custom(self):
        """Test NormalizeConfig with custom values."""
        config = NormalizeConfig(min=-1.0, max=1.0)
        assert config.min == -1.0
        assert config.max == 1.0


class TestMappingItem:
    """Tests for MappingItem class."""

    def test_mapping_item_basic(self):
        """Test basic MappingItem creation."""
        item = MappingItem(name="joint_0")
        assert item.name == "joint_0"
        assert item.source_name is None
        assert item.index is None
        assert item.index_range is None
        assert item.offset == 0.0
        assert item.inverted is False

    def test_mapping_item_with_index(self):
        """Test MappingItem with index."""
        item = MappingItem(name="joint_0", index=5)
        assert item.index == 5

    def test_mapping_item_with_index_range(self):
        """Test MappingItem with index_range."""
        index_range = IndexRangeConfig(start=0, end=3)
        item = MappingItem(name="pose", index_range=index_range)
        assert item.index_range == index_range

    def test_mapping_item_index_and_range_conflict(self):
        """Test MappingItem validation when both index and index_range are provided."""
        index_range = IndexRangeConfig(start=0, end=3)
        with pytest.raises(
            ValueError, match="index and index_range cannot be provided together"
        ):
            MappingItem(name="pose", index=5, index_range=index_range)


class TestOrientationConfig:
    """Tests for OrientationConfig class."""

    def test_orientation_config_defaults(self):
        """Test OrientationConfig with default values."""
        config = OrientationConfig()
        assert config.type == RotationConfig.QUATERNION
        assert config.quaternion_order == QuaternionOrderConfig.XYZW
        assert config.euler_order == EulerOrderConfig.XYZ
        assert config.angle_units == AngleConfig.RADIANS

    def test_orientation_config_euler(self):
        """Test OrientationConfig with euler type."""
        config = OrientationConfig(
            type=RotationConfig.EULER,
            euler_order=EulerOrderConfig.ZYX,
            angle_units=AngleConfig.DEGREES,
        )
        assert config.type == RotationConfig.EULER
        assert config.euler_order == EulerOrderConfig.ZYX
        assert config.angle_units == AngleConfig.DEGREES


class TestDataFormat:
    """Tests for DataFormat class."""

    def test_data_format_defaults(self):
        """Test DataFormat with default values."""
        fmt = DataFormat()
        assert fmt.image_convention == ImageConventionConfig.CHANNELS_LAST
        assert fmt.order_of_channels == ImageChannelOrderConfig.RGB
        assert fmt.normalized_pixel_values is False
        assert fmt.angle_units == AngleConfig.RADIANS
        assert fmt.torque_units == TorqueUnitsConfig.NM
        assert fmt.distance_units == DistanceUnitsConfig.M
        assert fmt.pose_type == PoseConfig.MATRIX
        assert fmt.orientation is None
        assert fmt.language_type == LanguageConfig.STRING
        assert fmt.normalize is None

    def test_data_format_custom(self):
        """Test DataFormat with custom values."""
        normalize = NormalizeConfig(min=0.0, max=1.0)
        orientation = OrientationConfig(type=RotationConfig.EULER)
        fmt = DataFormat(
            image_convention=ImageConventionConfig.CHANNELS_FIRST,
            order_of_channels=ImageChannelOrderConfig.BGR,
            normalized_pixel_values=True,
            angle_units=AngleConfig.DEGREES,
            torque_units=TorqueUnitsConfig.NCM,
            distance_units=DistanceUnitsConfig.MM,
            pose_type=PoseConfig.POSITION_ORIENTATION,
            orientation=orientation,
            language_type=LanguageConfig.BYTES,
            normalize=normalize,
        )
        assert fmt.image_convention == ImageConventionConfig.CHANNELS_FIRST
        assert fmt.order_of_channels == ImageChannelOrderConfig.BGR
        assert fmt.normalized_pixel_values is True
        assert fmt.angle_units == AngleConfig.DEGREES
        assert fmt.torque_units == TorqueUnitsConfig.NCM
        assert fmt.distance_units == DistanceUnitsConfig.MM
        assert fmt.pose_type == PoseConfig.POSITION_ORIENTATION
        assert fmt.orientation == orientation
        assert fmt.language_type == LanguageConfig.BYTES
        assert fmt.normalize == normalize


class TestDatasetConfig:
    """Tests for DatasetConfig class."""

    def test_dataset_config_basic(self):
        """Test basic DatasetConfig creation."""
        output_dataset = OutputDatasetConfig(name="output_dataset")
        robot = RobotConfig(name="test_robot")
        config = DatasetImportConfig(
            input_dataset_name="input_dataset",
            output_dataset=output_dataset,
            robot=robot,
        )
        assert config.input_dataset_name == "input_dataset"
        assert config.output_dataset == output_dataset
        assert config.robot == robot
        assert config.frequency is None
        assert config.data_import_config == {}

    def test_dataset_config_from_file_yaml(self, tmp_path):
        """Test DatasetConfig.from_file with YAML file"""
        config_path = tmp_path / "config.yaml"
        config_data = {
            "input_dataset_name": "input_dataset",
            "output_dataset": {"name": "output_dataset"},
            "robot": {"name": "test_robot"},
            "data_import_config": {
                "RGB_IMAGES": {
                    "source": "camera0",
                    "format": {
                        "image_convention": "CHANNELS_FIRST",
                        "order_of_channels": "RGB",
                        "normalized_pixel_values": False,
                    },
                    "mapping": [{
                        "name": "image",
                        "source_name": "camera0",
                    }],
                },
                "JOINT_POSITIONS": {
                    "source": "joint_sensor0",
                    "format": {
                        "angle_units": "RADIANS",
                    },
                    "mapping": [{
                        "name": "joint_0",
                        "index": 0,
                        "offset": 0.0,
                        "inverted": False,
                    }],
                },
            },
        }
        with config_path.open("w") as f:
            yaml.dump(config_data, f)

        config = DatasetImportConfig.from_file(config_path)
        assert config.input_dataset_name == "input_dataset"
        assert config.output_dataset.name == "output_dataset"
        assert config.robot.name == "test_robot"
        # Verify data_import_config contains the sample types
        assert DataType.RGB_IMAGES in config.data_import_config
        assert config.data_import_config[DataType.RGB_IMAGES].source == "camera0"
        assert DataType.JOINT_POSITIONS in config.data_import_config
        assert (
            config.data_import_config[DataType.JOINT_POSITIONS].source
            == "joint_sensor0"
        )
        # Check transforms
        rgb_transforms = (
            config.data_import_config[DataType.RGB_IMAGES]
            .mapping[0]
            .transforms.transforms
        )
        joint_transforms = (
            config.data_import_config[DataType.JOINT_POSITIONS]
            .mapping[0]
            .transforms.transforms
        )

        assert len(rgb_transforms) == 3
        assert len(joint_transforms) == 1

        assert isinstance(rgb_transforms[0], Clip)
        assert isinstance(rgb_transforms[1], CastToNumpyDtype)
        assert isinstance(rgb_transforms[2], ImageFormat)
        assert isinstance(joint_transforms[0], NumpyToScalar)

    def test_dataset_config_from_file_json(self, tmp_path):
        """Test DatasetConfig.from_file with JSON file."""
        config_path = tmp_path / "config.json"
        config_data = {
            "input_dataset_name": "input_dataset",
            "output_dataset": {"name": "output_dataset"},
            "robot": {"name": "test_robot"},
            "data_import_config": {
                "RGB_IMAGES": {
                    "source": "camera0",
                    "format": {
                        "image_convention": "CHANNELS_FIRST",
                        "order_of_channels": "RGB",
                        "normalized_pixel_values": False,
                    },
                    "mapping": [{
                        "name": "image",
                        "source_name": "camera0",
                    }],
                },
                "JOINT_POSITIONS": {
                    "source": "joint_sensor0",
                    "format": {
                        "angle_units": "RADIANS",
                    },
                    "mapping": [{
                        "name": "joint_0",
                        "index": 0,
                        "offset": 0.0,
                        "inverted": False,
                    }],
                },
            },
        }
        with config_path.open("w") as f:
            json.dump(config_data, f)

        config = DatasetImportConfig.from_file(config_path)
        assert config.input_dataset_name == "input_dataset"
        assert config.output_dataset.name == "output_dataset"
        assert config.robot.name == "test_robot"
        # Verify data_import_config contains the sample types
        assert DataType.RGB_IMAGES in config.data_import_config
        assert config.data_import_config[DataType.RGB_IMAGES].source == "camera0"
        assert DataType.JOINT_POSITIONS in config.data_import_config
        assert (
            config.data_import_config[DataType.JOINT_POSITIONS].source
            == "joint_sensor0"
        )
        # Check transforms
        rgb_transforms = (
            config.data_import_config[DataType.RGB_IMAGES]
            .mapping[0]
            .transforms.transforms
        )
        joint_transforms = (
            config.data_import_config[DataType.JOINT_POSITIONS]
            .mapping[0]
            .transforms.transforms
        )

        assert len(rgb_transforms) == 3
        assert len(joint_transforms) == 1

        assert isinstance(rgb_transforms[0], Clip)
        assert isinstance(rgb_transforms[1], CastToNumpyDtype)
        assert isinstance(rgb_transforms[2], ImageFormat)
        assert isinstance(joint_transforms[0], NumpyToScalar)

    def test_dataset_config_from_file_not_found(self, tmp_path):
        """Test DatasetConfig.from_file with non-existent file."""
        config_path = tmp_path / "missing_config.yaml"
        with pytest.raises(FileNotFoundError):
            DatasetImportConfig.from_file(config_path)

    def test_dataset_config_from_file_invalid_format(self, tmp_path):
        """Test DatasetConfig.from_file with invalid file format."""
        config_path = tmp_path / "config.txt"
        config_path.write_text("some content")
        with pytest.raises(RuntimeError, match="Unsupported config format"):
            DatasetImportConfig.from_file(config_path)

    def test_dataset_config_from_file_invalid_yaml(self, tmp_path):
        """Test DatasetConfig.from_file with invalid YAML."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("invalid: yaml: content: [")
        with pytest.raises(RuntimeError):
            DatasetImportConfig.from_file(config_path)

    def test_dataset_config_from_file_invalid_json(self, tmp_path):
        """Test DatasetConfig.from_file with invalid JSON."""
        config_path = tmp_path / "config.json"
        config_path.write_text("{invalid json}")
        with pytest.raises(RuntimeError):
            DatasetImportConfig.from_file(config_path)

    def test_dataset_config_with_data_points(self):
        """Test DatasetConfig with data points."""
        output_dataset = OutputDatasetConfig(name="output_dataset")
        robot = RobotConfig(name="test_robot")
        rgb_point = RGBCameraDataImportConfig(source="camera")
        joint_point = JointPositionsDataImportConfig(source="joints")
        data_import_config = {
            DataType.RGB_IMAGES: rgb_point,
            DataType.JOINT_POSITIONS: joint_point,
        }

        config = DatasetImportConfig(
            input_dataset_name="input_dataset",
            output_dataset=output_dataset,
            robot=robot,
            data_import_config=data_import_config,
        )
        assert config.data_import_config[DataType.RGB_IMAGES] == rgb_point
        assert config.data_import_config[DataType.JOINT_POSITIONS] == joint_point
