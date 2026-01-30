"""Tests for JointData and BatchedJointData."""

import pytest
import torch

from neuracore_types import BatchedJointData, JointData
from neuracore_types.batched_nc_data import DATA_TYPE_TO_BATCHED_NC_DATA_CLASS
from neuracore_types.importer.config import (
    AngleConfig,
    TorqueUnitsConfig,
    VisualJointTypeConfig,
)
from neuracore_types.importer.data_config import DataFormat, MappingItem
from neuracore_types.importer.transform import (
    Clip,
    DegreesToRadians,
    FlipSign,
    NumpyToScalar,
    Offset,
    Scale,
    Unnormalize,
)
from neuracore_types.nc_data import DATA_TYPE_TO_NC_DATA_CLASS, DataType
from neuracore_types.nc_data.joint_data import (
    JointPositionsDataImportConfig,
    JointTorquesDataImportConfig,
    JointVelocitiesDataImportConfig,
    VisualJointPositionsDataImportConfig,
)


class TestJointData:
    """Tests for JointData functionality."""

    def test_sample(self):
        """Test JointData.sample() creates valid instance."""
        data = JointData.sample()
        assert isinstance(data, JointData)
        assert isinstance(data.value, float)
        assert data.value == 0.0
        assert data.type == "JointData"

    def test_calculate_statistics(self):
        """Test calculate_statistics() for joint data."""
        data = JointData(value=1.5)
        stats = data.calculate_statistics()

        assert stats.type == "JointDataStats"
        assert stats.value is not None
        assert stats.value.mean[0] == 1.5
        assert stats.value.count[0] == 1

    def test_serialization(self):
        """Test JSON serialization and deserialization."""
        data = JointData(value=2.5)
        json_str = data.model_dump_json()
        loaded = JointData.model_validate_json(json_str)

        assert loaded.value == data.value
        assert loaded.timestamp == data.timestamp

    def test_invalid_type(self):
        """Test that invalid joint value type raises error."""
        with pytest.raises(Exception):
            JointData(value="not a number")

    def test_negative_value(self):
        """Test handling of negative joint values."""
        data = JointData(value=-1.5)
        assert data.value == -1.5

    def test_zero_value(self):
        """Test handling of zero joint value."""
        data = JointData(value=0.0)
        assert data.value == 0.0

    def test_large_value(self):
        """Test handling of large joint values."""
        data = JointData(value=1000.0)
        assert data.value == 1000.0


class TestBatchedJointData:
    """Tests for BatchedJointData functionality."""

    def test_from_nc_data(self):
        """Test BatchedJointData.from_nc_data() conversion."""
        joint_data = JointData(value=2.5)
        batched = BatchedJointData.from_nc_data(joint_data)

        assert isinstance(batched, BatchedJointData)
        assert batched.value.shape == (1, 1, 1)
        assert batched.value[0, 0, 0] == 2.5

    def test_sample(self):
        """Test BatchedJointData.sample() with different dimensions."""
        batched = BatchedJointData.sample(batch_size=4, time_steps=6)
        assert batched.value.shape == (4, 6, 1)

    def test_sample_single_dimension(self):
        """Test sample with single batch and timestep."""
        batched = BatchedJointData.sample(batch_size=1, time_steps=1)
        assert batched.value.shape == (1, 1, 1)

    def test_sample_large_dimensions(self):
        """Test sample with large dimensions."""
        batched = BatchedJointData.sample(batch_size=100, time_steps=50)
        assert batched.value.shape == (100, 50, 1)

    def test_to_device(self):
        """Test moving BatchedJointData to different device."""
        batched = BatchedJointData.sample(batch_size=2, time_steps=3)
        batched_cpu = batched.to(torch.device("cpu"))

        assert batched_cpu.value.device.type == "cpu"
        assert batched_cpu.value.shape == batched.value.shape

    def test_from_nc_data_preserves_value(self):
        """Test that from_nc_data preserves exact value."""
        test_value = 3.14159
        joint_data = JointData(value=test_value)
        batched = BatchedJointData.from_nc_data(joint_data)

        assert torch.isclose(
            batched.value[0, 0, 0], torch.tensor(test_value), rtol=1e-5
        )

    def test_can_serialize_deserialize(self):
        """Test JSON serialization and deserialization of BatchedJointData."""
        batched = BatchedJointData.sample(batch_size=2, time_steps=2)
        json_str = batched.model_dump_json()
        loaded = BatchedJointData.model_validate_json(json_str)

        assert torch.equal(loaded.value, batched.value)
        assert loaded.value.shape == batched.value.shape

    def test_from_nc_data_list_single_item(self):
        """Test from_nc_data_list with single joint value."""
        joint_data = JointData(value=1.5)
        batched = BatchedJointData.from_nc_data_list([joint_data])

        assert isinstance(batched, BatchedJointData)
        assert batched.value.shape == (1, 1, 1)
        assert batched.value[0, 0, 0] == 1.5

    def test_from_nc_data_list_multiple_items(self):
        """Test from_nc_data_list with multiple joint values."""
        values = [0.5, 1.0, 1.5, 2.0, 2.5]
        joint_data_list = [JointData(value=v) for v in values]
        batched = BatchedJointData.from_nc_data_list(joint_data_list)

        assert batched.value.shape == (1, 5, 1)
        for i, expected_val in enumerate(values):
            assert torch.isclose(
                batched.value[0, i, 0], torch.tensor(expected_val), rtol=1e-5
            )

    def test_from_nc_data_list_large_batch(self):
        """Test from_nc_data_list with large number of joint values."""
        num_joints = 100
        joint_data_list = [JointData(value=float(i)) for i in range(num_joints)]
        batched = BatchedJointData.from_nc_data_list(joint_data_list)

        assert batched.value.shape == (1, num_joints, 1)
        assert batched.value[0, 0, 0] == 0.0
        assert batched.value[0, 99, 0] == 99.0

    def test_from_nc_data_list_preserves_order(self):
        """Test that from_nc_data_list preserves order of joint values."""
        values = [3.14, -1.5, 0.0, 2.71, -5.0]
        joint_data_list = [JointData(value=v) for v in values]
        batched = BatchedJointData.from_nc_data_list(joint_data_list)

        for i, expected_val in enumerate(values):
            assert torch.isclose(
                batched.value[0, i, 0], torch.tensor(expected_val), rtol=1e-5
            )

    def test_from_nc_data_list_negative_values(self):
        """Test from_nc_data_list with negative joint values."""
        joint_data_list = [JointData(value=-1.0), JointData(value=-2.0)]
        batched = BatchedJointData.from_nc_data_list(joint_data_list)

        assert batched.value[0, 0, 0] == -1.0
        assert batched.value[0, 1, 0] == -2.0


class TestJointDataStatistics:
    """Tests for JointData statistics."""

    def test_statistics_values(self):
        """Test that statistics contain correct values."""
        data = JointData(value=5.0)
        stats = data.calculate_statistics()

        assert stats.value.mean[0] == 5.0
        assert stats.value.count[0] == 1
        assert stats.value.min[0] == 5.0
        assert stats.value.max[0] == 5.0
        assert stats.value.std[0] == 0.0

    def test_statistics_concatenation(self):
        """Test that joint statistics can be concatenated."""
        data1 = JointData(value=1.0)
        data2 = JointData(value=2.0)

        stats1 = data1.calculate_statistics()
        stats2 = data2.calculate_statistics()

        concatenated = stats1.value.concatenate(stats2.value)
        assert len(concatenated.mean) == 2
        assert concatenated.mean[0] == 1.0
        assert concatenated.mean[1] == 2.0

    def test_statistics_with_negative_value(self):
        """Test statistics with negative joint value."""
        data = JointData(value=-3.5)
        stats = data.calculate_statistics()

        assert stats.value.mean[0] == -3.5
        assert stats.value.min[0] == -3.5
        assert stats.value.max[0] == -3.5


class TestJointPositionsDataImportConfig:
    """Tests for JointPositionsDataImportConfig class."""

    def test_joint_positions_auto_index(self):
        """Test JointPositionsDataImportConfig auto-assigns indexes."""
        data_point = JointPositionsDataImportConfig(
            source="joints",
            mapping=[
                MappingItem(name="joint_0"),
                MappingItem(name="joint_1"),
                MappingItem(name="joint_2"),
            ],
        )
        assert data_point.mapping[0].index == 0
        assert data_point.mapping[1].index == 1
        assert data_point.mapping[2].index == 2

    def test_joint_positions_all_indexes_provided(self):
        """Test JointPositionsDataImportConfig with all indexes provided."""
        data_point = JointPositionsDataImportConfig(
            source="joints",
            mapping=[
                MappingItem(name="joint_0", index=5),
                MappingItem(name="joint_1", index=6),
            ],
        )
        assert data_point.mapping[0].index == 5
        assert data_point.mapping[1].index == 6

    def test_joint_positions_mixed_indexes_invalid(self):
        """Test JointPositionsDataImportConfig validation with mixed indexes."""
        with pytest.raises(
            ValueError,
            match="All or none of the mapping items in",
        ):
            JointPositionsDataImportConfig(
                source="joints",
                mapping=[
                    MappingItem(name="joint_0", index=0),
                    MappingItem(name="joint_1"),  # Missing index
                ],
            )

    def test_joint_positions_transforms_radians(self):
        """Test JointPositionsDataImportConfig transforms for radians."""
        data_point = JointPositionsDataImportConfig(
            source="joints",
            mapping=[MappingItem(name="joint_0")],
            format=DataFormat(angle_units=AngleConfig.RADIANS),
        )
        transforms = data_point.mapping[0].transforms.transforms
        assert not any(isinstance(t, DegreesToRadians) for t in transforms)
        assert isinstance(transforms[-1], NumpyToScalar)

    def test_joint_positions_transforms_degrees(self):
        """Test JointPositionsDataImportConfig transforms for degrees."""
        data_point = JointPositionsDataImportConfig(
            source="joints",
            mapping=[MappingItem(name="joint_0")],
            format=DataFormat(angle_units=AngleConfig.DEGREES),
        )
        transforms = data_point.mapping[0].transforms.transforms
        assert any(isinstance(t, DegreesToRadians) for t in transforms)

    def test_joint_positions_transforms_inverted(self):
        """Test JointPositionsDataImportConfig transforms with inverted flag."""
        data_point = JointPositionsDataImportConfig(
            source="joints",
            mapping=[MappingItem(name="joint_0", inverted=True)],
        )
        transforms = data_point.mapping[0].transforms.transforms
        assert any(isinstance(t, FlipSign) for t in transforms)

    def test_joint_positions_transforms_offset(self):
        """Test JointPositionsDataImportConfig transforms with offset."""
        data_point = JointPositionsDataImportConfig(
            source="joints",
            mapping=[MappingItem(name="joint_0", offset=1.5)],
        )
        transforms = data_point.mapping[0].transforms.transforms
        assert any(isinstance(t, Offset) for t in transforms)


class TestJointVelocitiesDataImportConfig:
    """Tests for JointVelocitiesDataImportConfig class."""

    def test_joint_velocities_auto_index(self):
        """Test JointVelocitiesDataImportConfig auto-assigns indexes."""
        data_point = JointVelocitiesDataImportConfig(
            source="joints",
            mapping=[MappingItem(name="joint_0"), MappingItem(name="joint_1")],
        )
        assert data_point.mapping[0].index == 0
        assert data_point.mapping[1].index == 1

    def test_joint_velocities_transforms_degrees(self):
        """Test JointVelocitiesDataImportConfig transforms for degrees."""
        data_point = JointVelocitiesDataImportConfig(
            source="joints",
            mapping=[MappingItem(name="joint_0")],
            format=DataFormat(angle_units=AngleConfig.DEGREES),
        )
        transforms = data_point.mapping[0].transforms.transforms
        assert any(isinstance(t, DegreesToRadians) for t in transforms)


class TestJointTorquesDataImportConfig:
    """Tests for JointTorquesDataImportConfig class."""

    def test_joint_torques_transforms_nm(self):
        """Test JointTorquesDataImportConfig transforms for Nm."""
        data_point = JointTorquesDataImportConfig(
            source="joints",
            mapping=[MappingItem(name="joint_0")],
            format=DataFormat(torque_units=TorqueUnitsConfig.NM),
        )
        transforms = data_point.mapping[0].transforms.transforms
        assert not any(isinstance(t, Scale) for t in transforms)

    def test_joint_torques_transforms_ncm(self):
        """Test JointTorquesDataImportConfig transforms for Ncm."""
        data_point = JointTorquesDataImportConfig(
            source="joints",
            mapping=[MappingItem(name="joint_0")],
            format=DataFormat(torque_units=TorqueUnitsConfig.NCM),
        )
        transforms = data_point.mapping[0].transforms.transforms
        assert any(isinstance(t, Scale) for t in transforms)


class TestVisualJointPositionsDataType:
    """Tests for VISUAL_JOINT_POSITIONS DataType mapping."""

    def test_visual_joint_positions_data_type_exists(self):
        """Test that VISUAL_JOINT_POSITIONS DataType exists."""
        assert hasattr(DataType, "VISUAL_JOINT_POSITIONS")
        assert DataType.VISUAL_JOINT_POSITIONS.value == "VISUAL_JOINT_POSITIONS"

    def test_visual_joint_positions_data_type_mapped_to_nc_data_class(self):
        """Test VISUAL_JOINT_POSITIONS mapped to JointData."""
        assert DataType.VISUAL_JOINT_POSITIONS in DATA_TYPE_TO_NC_DATA_CLASS
        assert DATA_TYPE_TO_NC_DATA_CLASS[DataType.VISUAL_JOINT_POSITIONS] == JointData

    def test_visual_joint_positions_data_type_mapped_to_batched_nc_data_class(self):
        """Test VISUAL_JOINT_POSITIONS mapped to BatchedJointData."""
        assert DataType.VISUAL_JOINT_POSITIONS in DATA_TYPE_TO_BATCHED_NC_DATA_CLASS
        assert (
            DATA_TYPE_TO_BATCHED_NC_DATA_CLASS[DataType.VISUAL_JOINT_POSITIONS]
            == BatchedJointData
        )

    def test_visual_joint_positions_uses_same_data_class_as_state(self):
        """Test that visual joint positions and state
        use the same underlying data classes."""
        assert (
            DATA_TYPE_TO_NC_DATA_CLASS[DataType.VISUAL_JOINT_POSITIONS]
            == DATA_TYPE_TO_NC_DATA_CLASS[DataType.JOINT_POSITIONS]
        )
        assert (
            DATA_TYPE_TO_BATCHED_NC_DATA_CLASS[DataType.VISUAL_JOINT_POSITIONS]
            == DATA_TYPE_TO_BATCHED_NC_DATA_CLASS[DataType.JOINT_POSITIONS]
        )


class TestJointTargetPositionsDataType:
    """Tests for JOINT_TARGET_POSITIONS DataType mapping."""

    def test_target_positions_data_type_exists(self):
        """Test that JOINT_TARGET_POSITIONS DataType exists."""
        assert hasattr(DataType, "JOINT_TARGET_POSITIONS")
        assert DataType.JOINT_TARGET_POSITIONS.value == "JOINT_TARGET_POSITIONS"

    def test_target_positions_data_type_mapped_to_nc_data_class(self):
        """Test that JOINT_TARGET_POSITIONS is mapped to JointData."""
        assert DataType.JOINT_TARGET_POSITIONS in DATA_TYPE_TO_NC_DATA_CLASS
        assert DATA_TYPE_TO_NC_DATA_CLASS[DataType.JOINT_TARGET_POSITIONS] == JointData

    def test_target_positions_data_type_mapped_to_batched_nc_data_class(self):
        """Test that JOINT_TARGET_POSITIONS is mapped to BatchedJointData."""
        assert DataType.JOINT_TARGET_POSITIONS in DATA_TYPE_TO_BATCHED_NC_DATA_CLASS
        assert (
            DATA_TYPE_TO_BATCHED_NC_DATA_CLASS[DataType.JOINT_TARGET_POSITIONS]
            == BatchedJointData
        )

    def test_target_uses_same_data_class_as_state(self):
        """Test that target and state use the same underlying data classes."""
        assert (
            DATA_TYPE_TO_NC_DATA_CLASS[DataType.JOINT_TARGET_POSITIONS]
            == DATA_TYPE_TO_NC_DATA_CLASS[DataType.JOINT_POSITIONS]
        )
        assert (
            DATA_TYPE_TO_BATCHED_NC_DATA_CLASS[DataType.JOINT_TARGET_POSITIONS]
            == DATA_TYPE_TO_BATCHED_NC_DATA_CLASS[DataType.JOINT_POSITIONS]
        )


class TestVisualJointPositionsDataImportConfig:
    """Tests for VisualJointPositionsDataImportConfig class."""

    def test_gripper_type_all_indexes_provided(self):
        """Test VisualJointPositionsDataImportConfig with GRIPPER type."""
        config = VisualJointPositionsDataImportConfig(
            source="gripper",
            format=DataFormat(visual_joint_type=VisualJointTypeConfig.GRIPPER),
            mapping=[
                MappingItem(name="gripper_joint_0", index=0),
                MappingItem(name="gripper_joint_1", index=1),
            ],
        )
        assert config.mapping[0].index == 0
        assert config.mapping[1].index == 1

    def test_gripper_type_missing_indexes_invalid(self):
        """Test validation with GRIPPER type missing indexes."""
        with pytest.raises(
            ValueError,
            match="All indexes must be provided for gripper visual joint positions",
        ):
            VisualJointPositionsDataImportConfig(
                source="gripper",
                format=DataFormat(visual_joint_type=VisualJointTypeConfig.GRIPPER),
                mapping=[
                    MappingItem(name="gripper_joint_0", index=0),
                    MappingItem(name="gripper_joint_1"),  # Missing index
                ],
            )

    def test_custom_type_auto_index(self):
        """Test CUSTOM type auto-assigns indexes."""
        config = VisualJointPositionsDataImportConfig(
            source="joints",
            format=DataFormat(visual_joint_type=VisualJointTypeConfig.CUSTOM),
            mapping=[
                MappingItem(name="joint_0"),
                MappingItem(name="joint_1"),
                MappingItem(name="joint_2"),
            ],
        )
        assert config.mapping[0].index == 0
        assert config.mapping[1].index == 1
        assert config.mapping[2].index == 2

    def test_custom_type_all_indexes_provided(self):
        """Test CUSTOM type with all indexes provided."""
        config = VisualJointPositionsDataImportConfig(
            source="joints",
            format=DataFormat(visual_joint_type=VisualJointTypeConfig.CUSTOM),
            mapping=[
                MappingItem(name="joint_0", index=5),
                MappingItem(name="joint_1", index=6),
            ],
        )
        assert config.mapping[0].index == 5
        assert config.mapping[1].index == 6

    def test_custom_type_mixed_indexes_invalid(self):
        """Test validation with CUSTOM type and mixed indexes."""
        with pytest.raises(
            ValueError,
            match="All or none of the mapping items in",
        ):
            VisualJointPositionsDataImportConfig(
                source="joints",
                format=DataFormat(visual_joint_type=VisualJointTypeConfig.CUSTOM),
                mapping=[
                    MappingItem(name="joint_0", index=0),
                    MappingItem(name="joint_1"),  # Missing index
                ],
            )

    def test_gripper_type_transforms(self):
        """Test VisualJointPositionsDataImportConfig transforms for GRIPPER type."""
        config = VisualJointPositionsDataImportConfig(
            source="gripper",
            format=DataFormat(visual_joint_type=VisualJointTypeConfig.GRIPPER),
            mapping=[MappingItem(name="gripper_joint_0", index=0)],
        )
        transforms = config.mapping[0].transforms.transforms
        # Should have Clip, Unnormalize, and NumpyToScalar
        assert any(isinstance(t, Clip) for t in transforms)
        assert any(isinstance(t, Unnormalize) for t in transforms)
        assert isinstance(transforms[-1], NumpyToScalar)

    def test_gripper_type_transforms_with_inverted(self):
        """Test GRIPPER type transforms with inverted flag."""
        config = VisualJointPositionsDataImportConfig(
            source="gripper",
            format=DataFormat(visual_joint_type=VisualJointTypeConfig.GRIPPER),
            mapping=[MappingItem(name="gripper_joint_0", index=0, inverted=True)],
        )
        transforms = config.mapping[0].transforms.transforms
        assert any(isinstance(t, Clip) for t in transforms)
        assert any(isinstance(t, Unnormalize) for t in transforms)
        assert any(isinstance(t, FlipSign) for t in transforms)
        assert isinstance(transforms[-1], NumpyToScalar)

    def test_gripper_type_transforms_with_offset(self):
        """Test GRIPPER type transforms with offset."""
        config = VisualJointPositionsDataImportConfig(
            source="gripper",
            format=DataFormat(visual_joint_type=VisualJointTypeConfig.GRIPPER),
            mapping=[MappingItem(name="gripper_joint_0", index=0, offset=1.5)],
        )
        transforms = config.mapping[0].transforms.transforms
        assert any(isinstance(t, Clip) for t in transforms)
        assert any(isinstance(t, Unnormalize) for t in transforms)
        assert any(isinstance(t, Offset) for t in transforms)
        assert isinstance(transforms[-1], NumpyToScalar)

    def test_custom_type_transforms_degrees_inverted_offset(self):
        """Test CUSTOM type transforms with degrees, inverted, and offset."""
        config = VisualJointPositionsDataImportConfig(
            source="joints",
            format=DataFormat(
                visual_joint_type=VisualJointTypeConfig.CUSTOM,
                angle_units=AngleConfig.DEGREES,
            ),
            mapping=[
                MappingItem(
                    name="joint_0",
                    inverted=True,
                    offset=2.0,
                )
            ],
        )
        transforms = config.mapping[0].transforms.transforms
        assert any(isinstance(t, DegreesToRadians) for t in transforms)
        assert any(isinstance(t, FlipSign) for t in transforms)
        assert any(isinstance(t, Offset) for t in transforms)
        assert isinstance(transforms[-1], NumpyToScalar)

    def test_gripper_type_multiple_joints(self):
        """Test GRIPPER type with multiple joints."""
        config = VisualJointPositionsDataImportConfig(
            source="gripper",
            format=DataFormat(visual_joint_type=VisualJointTypeConfig.GRIPPER),
            mapping=[
                MappingItem(name="gripper_joint_0", index=0),
                MappingItem(name="gripper_joint_1", index=1),
                MappingItem(name="gripper_joint_2", index=2),
            ],
        )
        assert len(config.mapping) == 3
        for i, item in enumerate(config.mapping):
            assert item.index == i
            transforms = item.transforms.transforms
            assert any(isinstance(t, Clip) for t in transforms)
            assert any(isinstance(t, Unnormalize) for t in transforms)
            assert isinstance(transforms[-1], NumpyToScalar)
