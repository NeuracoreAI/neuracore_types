"""Defines the core data structures used throughout Neuracore."""

import base64
import time
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Annotated,
    Any,
    Dict,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)
from uuid import uuid4

import numpy as np
from PIL.Image import Image
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    field_serializer,
    field_validator,
)


def _sort_dict_by_keys(data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Sort a dictionary by its keys to ensure consistent ordering.

    This is a helper function used internally by the data models to ensure
    consistent dictionary ordering. Use the model's order() or
    sort_in_place() methods instead of calling this directly.

    Args:
        data_dict: Dictionary to sort, or None

    Returns:
        New dictionary with keys sorted alphabetically, or None if input was None
    """
    return {key: data_dict[key] for key in sorted(data_dict.keys())}


class DataType(str, Enum):
    """Enumeration of supported data types in the Neuracore system.

    Defines the standard data categories used for dataset organization,
    model training, and data processing pipelines.
    """

    # Robot state
    JOINT_POSITIONS = "JOINT_POSITIONS"
    JOINT_VELOCITIES = "JOINT_VELOCITIES"
    JOINT_TORQUES = "JOINT_TORQUES"
    JOINT_TARGET_POSITIONS = "JOINT_TARGET_POSITIONS"
    END_EFFECTOR_POSES = "END_EFFECTOR_POSES"
    PARALLEL_GRIPPER_OPEN_AMOUNTS = "PARALLEL_GRIPPER_OPEN_AMOUNTS"

    # Vision
    RGB_IMAGES = "RGB_IMAGES"
    DEPTH_IMAGES = "DEPTH_IMAGES"
    POINT_CLOUDS = "POINT_CLOUDS"

    # Other
    POSES = "POSES"
    LANGUAGE = "LANGUAGE"
    CUSTOM = "CUSTOM"

    def from_numpy(self, array: np.ndarray) -> "NCData":
        """Create a sample NCData instance for this DataType.

        Returns:
            An instance of the corresponding NCData subclass.
        """
        mapping = {
            DataType.JOINT_POSITIONS: JointData.from_numpy,
            DataType.JOINT_VELOCITIES: JointData.from_numpy,
            DataType.JOINT_TORQUES: JointData.from_numpy,
            DataType.JOINT_TARGET_POSITIONS: JointData.from_numpy,
            DataType.END_EFFECTOR_POSES: EndEffectorPoseData.from_numpy,
            DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS: (
                ParallelGripperOpenAmountData.from_numpy
            ),
            DataType.POSES: PoseData.from_numpy,
            DataType.POINT_CLOUDS: PointCloudData.from_numpy,
            # LANGUAGE and CUSTOM types are not supported for from_numpy
        }
        if self not in mapping:
            raise NotImplementedError(f"from_numpy not implemented for {self.value}")
        return mapping[self](array)


class NCData(BaseModel):
    """Base class for all Neuracore data with automatic timestamping.

    Provides a common base for all data types in the system with automatic
    timestamp generation for temporal synchronization and data ordering.
    """

    timestamp: float = Field(default_factory=lambda: time.time())

    def order(self) -> "NCData":
        """Return a new instance with sorted data.

        This method should be overridden by subclasses to implement specific
        ordering logic for the data type. The base class implementation does
        nothing and returns self.
        """
        return self

    def numpy(self) -> np.ndarray:
        """Convert the data to a NumPy array.

        Returns:
            NumPy array representation of the data.
        """
        raise NotImplementedError("Subclasses must implement numpy() method.")

    @staticmethod
    def from_numpy(array: np.ndarray) -> "NCData":
        """Create an NCData instance from a NumPy array.

        Args:
            array: NumPy array to convert.

        Returns:
            NCData instance created from the array.
        """
        raise NotImplementedError("Subclasses must implement from_numpy() method.")


class JointData(NCData):
    """Robot joint state data including positions, velocities, or torques.

    Represents joint-space data for robotic systems with support for named
    joints and additional auxiliary values. Used for positions, velocities,
    torques, and target positions.
    """

    type: Literal["JointData"] = "JointData"
    values: dict[str, float]

    def order(self) -> "JointData":
        """Return a new JointData instance with sorted joint names.

        Returns:
            New JointData with alphabetically sorted joint names.
        """
        return JointData(
            timestamp=self.timestamp,
            values=_sort_dict_by_keys(self.values),
        )

    def numpy(self) -> np.ndarray:
        """Convert the joint values to a NumPy array.

        Returns:
            NumPy array of joint values.
        """
        return np.array(list(self.values.values()), dtype=np.float32)

    @staticmethod
    def from_numpy(array: np.ndarray) -> "JointData":
        """Create a JointData instance from a NumPy array.

        Args:
            array: NumPy array to convert.

        Returns:
            JointData instance created from the array.
        """
        values = {f"joint{i}": float(val) for i, val in enumerate(array)}
        return JointData(values=values)

    def __getitem__(self, key: str) -> float:
        """Get item by joint name."""
        return self.values[key]

    def __setitem__(self, key: str, value: float) -> None:
        """Set item by joint name."""
        self.values[key] = value

    def __len__(self) -> int:
        """Get the number of joints."""
        return len(self.values)

    def keys(self):
        """Get the joint names."""
        return self.values.keys()


class CameraData(NCData):
    """Camera sensor data including images and calibration information.

    Contains image data along with camera intrinsic and extrinsic parameters
    for 3D reconstruction and computer vision applications. The frame field
    is populated during dataset iteration for efficiency.
    """

    type: Literal["CameraData"] = "CameraData"
    frame_idx: int = 0  # Needed so we can index video after sync
    extrinsics: Optional[list[list[float]]] = None
    intrinsics: Optional[list[list[float]]] = None
    frame: Optional[Union[Any, str]] = None  # Only filled in when using dataset iter

    def numpy(self) -> np.ndarray:
        """Convert the joint values to a NumPy array.

        Returns:
            NumPy array of joint values.
        """
        assert self.frame is not None, "Camera frame data is not available."
        assert isinstance(self.frame, Image), "Camera frame is not a PIL Image."
        return np.array(self.frame, dtype=np.float32)

    @staticmethod
    def from_numpy(array: np.ndarray) -> "CameraData":
        """Create a CameraData instance from a NumPy array.

        Args:
            array: NumPy array to convert.

        Returns:
            CameraData instance created from the array.
        """
        from PIL import Image as PILImage

        frame = PILImage.fromarray(array.astype(np.uint8))
        return CameraData(frame=frame)


class PoseData(NCData):
    """6DOF pose data for objects, end-effectors, or coordinate frames.

    Represents position and orientation information for tracking objects
    or robot components in 3D space. Poses are stored as dictionaries
    mapping pose names to [x, y, z, rx, ry, rz] values.
    """

    type: Literal["PoseData"] = "PoseData"
    pose: list[float]

    def numpy(self) -> np.ndarray:
        """Convert the pose values to a NumPy array.

        Returns:
            NumPy array of pose values.
        """
        return np.array(list(self.pose.values()), dtype=np.float32)

    @staticmethod
    def from_numpy(array: np.ndarray) -> "PoseData":
        """Create a PoseData instance from a NumPy array.

        Args:
            array: NumPy array to convert.

        Returns:
            PoseData instance created from the array.
        """
        pose = {f"pose{i}": list(array[i]) for i in range(len(array))}
        return PoseData(pose=pose)


class EndEffectorPoseData(NCData):
    """End-effector pose data.

    Contains the pose of end-effectors as a 7-element list containing the
    position and unit quaternion orientation [x, y, z, qx, qy, qz, qw].
    """

    type: Literal["EndEffectorPoseData"] = "EndEffectorPoseData"
    pose: list[float]

    def numpy(self) -> np.ndarray:
        """Convert the end-effector pose values to a NumPy array.

        Returns:
            NumPy array of end-effector pose values.
        """
        return np.array(list(self.poses.values()), dtype=np.float32)

    @staticmethod
    def from_numpy(array: np.ndarray) -> "EndEffectorPoseData":
        """Create an EndEffectorPoseData instance from a NumPy array.

        Args:
            array: NumPy array to convert.

        Returns:
            EndEffectorPoseData instance created from the array.
        """
        poses = {f"effector{i}": list(array[i]) for i in range(len(array))}
        return EndEffectorPoseData(poses=poses)


class ParallelGripperOpenAmountData(NCData):
    """Open amount data for parallel end effector gripper.

    Contains the state of parallel gripper opening amounts.
    """

    type: Literal["ParallelGripperOpenAmountData"] = "ParallelGripperOpenAmountData"
    open_amounts: float

    def numpy(self) -> np.ndarray:
        """Convert the gripper open amount values to a NumPy array.

        Returns:
            NumPy array of gripper open amount values.
        """
        return np.array(list(self.open_amounts.values()), dtype=np.float32)

    @staticmethod
    def from_numpy(array: np.ndarray) -> "ParallelGripperOpenAmountData":
        """Create a ParallelGripperOpenAmountData instance from a NumPy array.

        Args:
            array: NumPy array to convert.

        Returns:
            ParallelGripperOpenAmountData instance created from the array.
        """
        open_amounts = {f"gripper{i}": float(val) for i, val in enumerate(array)}
        return ParallelGripperOpenAmountData(open_amounts=open_amounts)


class PointCloudData(NCData):
    """3D point cloud data with optional RGB colouring and camera parameters.

    Represents 3D spatial data from depth sensors or LiDAR systems with
    optional colour information and camera calibration for registration.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["PointCloudData"] = "PointCloudData"
    points: Optional[np.ndarray] = None  # (N, 3) float16
    rgb_points: Optional[np.ndarray] = None  # (N, 3) uint8
    extrinsics: Optional[np.ndarray] = None  # (4, 4) float16
    intrinsics: Optional[np.ndarray] = None  # (3, 3) float16

    @staticmethod
    def _encode(arr: np.ndarray, dtype: Any) -> str:
        return base64.b64encode(arr.astype(dtype).tobytes()).decode("utf-8")

    @staticmethod
    def _decode(data: str, dtype: Any, shape: Tuple[int, ...]) -> np.ndarray:
        return np.frombuffer(
            base64.b64decode(data.encode("utf-8")), dtype=dtype
        ).reshape(*shape)

    @field_validator("points", mode="before")
    @classmethod
    def decode_points(cls, v: Union[str, np.ndarray]) -> Optional[np.ndarray]:
        """Decode base64 string to NumPy array if needed.

        Args:
            v: Base64 encoded string or NumPy array

        Returns:
            Decoded NumPy array or None
        """
        return cls._decode(v, np.float16, (-1, 3)) if isinstance(v, str) else v

    @field_validator("rgb_points", mode="before")
    @classmethod
    def decode_rgb_points(cls, v: Union[str, np.ndarray]) -> Optional[np.ndarray]:
        """Decode base64 string to NumPy array if needed.

        Args:
            v: Base64 encoded string or NumPy array

        Returns:
            Decoded NumPy array or None
        """
        return cls._decode(v, np.uint8, (-1, 3)) if isinstance(v, str) else v

    @field_validator("extrinsics", mode="before")
    @classmethod
    def decode_extrinsics(cls, v: Union[str, np.ndarray]) -> Optional[np.ndarray]:
        """Decode base64 string to NumPy array if needed.

        Args:
            v: Base64 encoded string or NumPy array

        Returns:
            Decoded NumPy array or None
        """
        return cls._decode(v, np.float16, (4, 4)) if isinstance(v, str) else v

    @field_validator("intrinsics", mode="before")
    @classmethod
    def decode_intrinsics(cls, v: Union[str, np.ndarray]) -> Optional[np.ndarray]:
        """Decode base64 string to NumPy array if needed.

        Args:
            v: Base64 encoded string or NumPy array

        Returns:
            Decoded NumPy array or None
        """
        return cls._decode(v, np.float16, (3, 3)) if isinstance(v, str) else v

    # --- Serializers (encode on dump) ---
    @field_serializer("points", when_used="json")
    def serialize_points(self, v: Optional[np.ndarray]) -> Optional[str]:
        """Encode NumPy array to base64 string if needed.

        Args:
            v: NumPy array to encode

        Returns:
            Base64 encoded string or None
        """
        return self._encode(v, np.float16) if v is not None else None

    @field_serializer("rgb_points", when_used="json")
    def serialize_rgb_points(self, v: Optional[np.ndarray]) -> Optional[str]:
        """Encode NumPy array to base64 string if needed.

        Args:
            v: NumPy array to encode

        Returns:
            Base64 encoded string or None
        """
        return self._encode(v, np.uint8) if v is not None else None

    @field_serializer("extrinsics", when_used="json")
    def serialize_extrinsics(self, v: Optional[np.ndarray]) -> Optional[str]:
        """Encode NumPy array to base64 string if needed.

        Args:
            v: NumPy array to encode

        Returns:
            Base64 encoded string or None
        """
        return self._encode(v, np.float16) if v is not None else None

    @field_serializer("intrinsics", when_used="json")
    def serialize_intrinsics(self, v: Optional[np.ndarray]) -> Optional[str]:
        """Encode NumPy array to base64 string if needed.

        Args:
            v: NumPy array to encode

        Returns:
            Base64 encoded string or None
        """
        return self._encode(v, np.float16) if v is not None else None

    def numpy(self) -> np.ndarray:
        """Convert the point cloud points to a NumPy array.

        Returns:
            NumPy array of point cloud points.
        """
        if self.points is None:
            raise ValueError("Point cloud data is not available.")
        return self.points

    @staticmethod
    def from_numpy(array: np.ndarray) -> "PointCloudData":
        """Create a PointCloudData instance from a NumPy array.

        Args:
            array: NumPy array to convert.

        Returns:
            PointCloudData instance created from the array.
        """
        return PointCloudData(points=array)


class LanguageData(NCData):
    """Natural language instruction or description data.

    Contains text-based information such as task descriptions, voice commands,
    or other linguistic data associated with robot demonstrations.
    """

    type: Literal["LanguageData"] = "LanguageData"
    text: str


class CustomData(NCData):
    """Generic container for application-specific data types.

    Provides a flexible way to include custom sensor data or application-specific
    information that doesn't fit into the standard data categories.
    """

    type: Literal["CustomData"] = "CustomData"
    data: Any


# Create a discriminated union type for all NCData subclasses
NCDataUnion = Annotated[
    Union[
        JointData,
        CameraData,
        PoseData,
        EndEffectorPoseData,
        ParallelGripperOpenAmountData,
        PointCloudData,
        LanguageData,
        CustomData,
    ],
    Field(discriminator="type"),
]


class SynchronizedPoint(BaseModel):
    """Synchronized collection of all sensor data at a single time point.

    Represents a complete snapshot of robot state and sensor information
    at a specific timestamp. Used for creating temporally aligned datasets
    and ensuring consistent data relationships across different sensors.
    """

    timestamp: float = Field(default_factory=lambda: time.time())
    robot_id: Optional[str] = None
    data: dict[DataType, Mapping[str, NCDataUnion]] = Field(default_factory=dict)

    def order(self) -> "SynchronizedPoint":
        """Return a new SynchronizedPoint with all dictionary data ordered."""
        return SynchronizedPoint(
            timestamp=self.timestamp,
            robot_id=self.robot_id,
            data={
                dt: {
                    name: data_item.order()
                    for name, data_item in _sort_dict_by_keys(data_dict).items()
                }
                for dt, data_dict in self.data.items()
            },
        )

    def __getitem__(self, key: Union[DataType, str]) -> Mapping[str, NCDataUnion]:
        """Get item by DataType or field name."""
        # If key is a DataType enum, access the nested data dict
        if isinstance(key, DataType):
            return self.data[key]
        # Otherwise, fall back to default Pydantic behavior for field names
        return super().__getitem__(key)

    def __setitem__(
        self, key: Union[DataType, str], value: Mapping[str, NCDataUnion]
    ) -> None:
        """Set item by DataType or field name."""
        # Same for setting
        if isinstance(key, DataType):
            self.data[key] = value
        else:
            super().__setitem__(key, value)


class SynchronizedEpisode(BaseModel):
    """Synchronized episode of time-ordered synchronized observations."""

    observations: list[SynchronizedPoint]
    start_time: float
    end_time: float
    robot_id: str

    def order(self) -> "SynchronizedEpisode":
        """Return a new SynchronizedEpisode with all synchronized observations ordered.

        Returns:
            New SynchronizedEpisode with all synchronized observations ordered.
        """
        return SynchronizedEpisode(
            observations=[observation.order() for observation in self.observations],
            start_time=self.start_time,
            end_time=self.end_time,
            robot_id=self.robot_id,
        )


class DataItemStats(BaseModel):
    """Statistical summary of data dimensions and distributions.

    Contains statistical information about data arrays including means,
    standard deviations, counts, and maximum lengths for normalization
    and model configuration purposes.

    Attributes:
        mean: List of means for each data dimension
        std: List of standard deviations for each data dimension
        count: List of counts for each data dimension
        min: List of minimum values for each data dimension
        max: List of maximum values for each data dimension
        max_len: Maximum length of the data arrays
        robot_to_ncdata_keys: Mapping of robot ids to their associated
            data keys for this data type
    """

    mean: list[float] = Field(default_factory=list)
    std: list[float] = Field(default_factory=list)
    count: list[int] = Field(default_factory=list)
    min: list[float] = Field(default_factory=list)
    max: list[float] = Field(default_factory=list)
    max_len: int = Field(default_factory=lambda data: len(data["mean"]))
    robot_to_ncdata_keys: dict[str, list[str]] = Field(default_factory=dict)


class DatasetStatistics(BaseModel):
    """Comprehensive description of dataset contents and statistics.

    Provides metadata about a complete dataset including statistical summaries
    for all data types, maximum counts for variable-length data, and methods
    for determining which data types are present.
    """

    total_num_transitions: int = 0
    data: dict[DataType, dict[str, DataItemStats]] = Field(default_factory=dict)

    def get_data_types(self) -> list[DataType]:
        """Determine which data types are present in the dataset.

        Analyzes the dataset statistics to identify which data modalities
        contain actual data (non-zero lengths/counts).

        Returns:
            List of DataType enums representing the data modalities
            present in this dataset.
        """
        return list(self.data.keys())

    def combine_for_data_type(self, data_type: DataType) -> DataItemStats:
        """Combine DataItemStats for a specific DataType across all entries.

        Args:
            data_type: The DataType for which to combine statistics.

        Returns:
            Combined DataItemStats for the specified DataType.
        """
        if data_type not in self.data:
            raise ValueError(f"DataType {data_type} not found in dataset statistics.")

        combined_stats = DataItemStats()
        for stats in self.data[data_type].values():
            combined_stats.mean.extend(stats.mean)
            combined_stats.std.extend(stats.std)
            combined_stats.count.extend(stats.count)
            combined_stats.min.extend(stats.min)
            combined_stats.max.extend(stats.max)
            combined_stats.max_len = max(combined_stats.max_len, stats.max_len)
            for robot_id, keys in stats.robot_to_ncdata_keys.items():
                if robot_id not in combined_stats.robot_to_ncdata_keys:
                    combined_stats.robot_to_ncdata_keys[robot_id] = []
                combined_stats.robot_to_ncdata_keys[robot_id].extend(keys)

        return combined_stats


class EpisodeStatistics(BaseModel):
    """Description of a single episode with statistics and counts.

    Provides metadata about an individual episode including data statistics,
    sensor counts, and episode length for analysis and processing.
    """

    # Episode metadata
    episode_length: int = 0
    data: dict[DataType, dict[str, DataItemStats]] = Field(default_factory=dict)

    def get_data_types(self) -> list[DataType]:
        """Determine which data types are present in the recording.

        Analyzes the recording statistics to identify which data modalities
        contain actual data (non-zero lengths/counts).

        Returns:
            List of DataType enums representing the data modalities
            present in this recording.
        """
        return list(self.data.keys())


class ModelInitDescription(BaseModel):
    """Configuration specification for initializing Neuracore models.

    Defines the model architecture requirements including dataset characteristics,
    input/output data types, and prediction horizons for model initialization
    and training configuration.
    """

    dataset_statistics: DatasetStatistics
    input_data_types: list[DataType]
    output_data_types: list[DataType]
    output_prediction_horizon: int = 1


class ModelPrediction(BaseModel):
    """Model inference output containing predictions and timing information.

    Represents the results of model inference including predicted outputs
    for each configured data type and optional timing information for
    performance monitoring.
    """

    outputs: dict[DataType, Any] = Field(default_factory=dict)
    prediction_time: Optional[float] = None


class PredictRequest(BaseModel):
    """Request model for server policy inference.

    Attributes:
        sync_point: The current observation.
        robot_name: The name of the robot to infer the policy for.
    """

    sync_point: SynchronizedPoint
    robot_name: Optional[str] = None


class SynchronizedDataset(BaseModel):
    """Represents a synchronized dataset of episodes.

    A Synchronized dataset groups related robot demonstrations together
    and maintains metadata about the collection as a whole.

    Attributes:
        id: Unique identifier for the synced dataset.
        parent_id: Unique identifier of the corresponding dataset.
        freq: Frequency at which dataset was processed.
        name: Human-readable name for the dataset.
        created_at: Unix timestamp of dataset creation.
        modified_at: Unix timestamp of last modification.
        description: Optional description of the dataset.
        recording_ids: List of recording IDs in this dataset
        num_demonstrations: Total number of demonstrations.
        total_duration_seconds: Total duration of all demonstrations.
        is_shared: Whether the dataset is shared with other users.
        metadata: Additional arbitrary metadata.
    """

    id: str
    parent_id: str
    freq: int
    name: str
    created_at: float
    modified_at: float
    description: Optional[str] = None
    recording_ids: list[str] = Field(default_factory=list)
    num_demonstrations: int = 0
    num_processed_demonstrations: int = 0
    total_duration_seconds: float = 0.0
    is_shared: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    dataset_statistics: DatasetStatistics = Field(default_factory=DatasetStatistics)
    all_data_types: dict[DataType, int] = Field(default_factory=dict)
    common_data_types: dict[DataType, int] = Field(default_factory=dict)


class Dataset(BaseModel):
    """Represents a dataset of unsynchronized episodes.

    Attributes:
        id: Unique identifier for the dataset.
        name: Human-readable name for the dataset.
        created_at: Unix timestamp of dataset creation.
        modified_at: Unix timestamp of last modification.
        description: Optional description of the dataset.
        tags: List of tags for categorizing the dataset.
        recording_ids: List of recording IDs in this dataset
        demonstration_ids: List of demonstration IDs in this dataset.
        num_demonstrations: Total number of demonstrations.
        total_duration_seconds: Total duration of all demonstrations.
        size_bytes: Total size of all demonstrations.
        is_shared: Whether the dataset is shared with other users.
        metadata: Additional arbitrary metadata.
        synced_dataset_ids: List of synced dataset IDs in this dataset.
                            They point to synced datasets that synchronized
                            this dataset at a particular frequency.
    """

    id: str
    name: str
    created_at: float
    modified_at: float
    description: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    recording_ids: list[str] = Field(default_factory=list)
    num_demonstrations: int = 0
    total_duration_seconds: float = 0.0
    size_bytes: int = 0
    is_shared: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    synced_dataset_ids: dict[str, Any] = Field(default_factory=dict)
    all_data_types: dict[DataType, int] = Field(default_factory=dict)
    common_data_types: dict[DataType, int] = Field(default_factory=dict)
    recording_ids_in_bucket: bool = False


class MessageType(str, Enum):
    """Enumerates the types of signaling messages for WebRTC handshakes.

    These types are used to identify the purpose of a message sent through
    the signaling server during connection establishment.
    """

    # Session Description Protocol (SDP) offer from the caller
    SDP_OFFER = "SDP_OFFER"

    # Session Description Protocol (SDP) answer from the callee
    SDP_ANSWER = "SDP_ANSWER"

    # Interactive Connectivity Establishment (ICE) candidate
    ICE_CANDIDATE = "ICE_CANDIDATE"

    # Request to open a new connection
    OPEN_CONNECTION = "OPEN_CONNECTION"


class HandshakeMessage(BaseModel):
    """Represents a signaling message for the WebRTC handshake process.

    This message is exchanged between two peers via a signaling server to
    negotiate the connection details, such as SDP offers/answers and ICE
    candidates.

    Attributes:
        from_id: The unique identifier of the sender peer.
        to_id: The unique identifier of the recipient peer.
        data: The payload of the message, typically an SDP string or a JSON
              object with ICE candidate information.
        connection_id: The unique identifier for the connection session.
        type: The type of the handshake message, as defined by MessageType.
        id: A unique identifier for the message itself.
    """

    from_id: str
    to_id: str
    data: str
    connection_id: str
    type: MessageType
    id: str = Field(default_factory=lambda: uuid4().hex)


class VideoFormat(str, Enum):
    """Enumerates video format styles over a WebRTC connection."""

    # use a standard video track with negotiated codec this is more efficient
    WEB_RTC_NEGOTIATED = "WEB_RTC_NEGOTIATED"
    # uses neuracore's data URI format over a custom data channel
    NEURACORE_CUSTOM = "NEURACORE_CUSTOM"


class OpenConnectionRequest(BaseModel):
    """Represents a request to open a new WebRTC connection.

    Attributes:
        from_id: The unique identifier of the consumer peer.
        to_id: The unique identifier of the producer peer.
        robot_id: The unique identifier for the robot to be created.
        robot_instance: The identifier for the instance of the robot to connect to.
        video_format: The type of video the consumer expects to receive.
        id: the identifier for this connection request.
        created_at: when the request was created.
    """

    from_id: str
    to_id: str
    robot_id: str
    robot_instance: NonNegativeInt
    video_format: VideoFormat
    id: str = Field(default_factory=lambda: uuid4().hex)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class OpenConnectionDetails(BaseModel):
    """The details describing properties about the new connection.

    Attributes:
        connection_token: The token used for security to establish the connection.
        robot_id: The unique identifier for the robot to connect to
        robot_instance: The identifier for the instance of the robot to connect to.
        video_format: The type of video the consumer expects to receive.
    """

    connection_token: str
    robot_id: str
    robot_instance: NonNegativeInt
    video_format: VideoFormat


class StreamAliveResponse(BaseModel):
    """Represents the response from asserting a stream is alive.

    This is returned when a client pings a stream to keep it active.

    Attributes:
        resurrected: A boolean indicating if the stream was considered dead
                     and has been successfully resurrected by this request.
    """

    resurrected: bool


class RobotInstanceIdentifier(NamedTuple):
    """A tuple that uniquely identifies a robot instance.

    Attributes:
        robot_id: The unique identifier of the robot providing the stream.
        robot_instance: The specific instance number of the robot.
    """

    robot_id: str
    robot_instance: int


class RobotStreamTrack(BaseModel):
    """Metadata for a robot's media stream track.

    This model holds all the necessary information to identify and manage
    a single media track (e.g., a video or audio feed) from a specific
    robot instance.

    Attributes:
        robot_id: The unique identifier of the robot providing the stream.
        robot_instance: The specific instance number of the robot.
        stream_id: The identifier for the overall media stream session.
        data_type: The type of media track.
        label: A human-readable label for the track (e.g., 'front_camera').
        mid: The media ID used in SDP, essential for WebRTC negotiation.
        id: A unique identifier for this track metadata object.
        created_at: The UTC timestamp when this track metadata was created.
    """

    robot_id: str
    robot_instance: NonNegativeInt
    stream_id: str
    data_type: DataType
    label: str
    mid: str
    id: str = Field(default_factory=lambda: uuid4().hex)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AvailableRobotInstance(BaseModel):
    """Represents a single, available instance of a robot.

    Attributes:
        robot_instance: The unique identifier for this robot instance.
        tracks: A dictionary of available media stream tracks for this instance.
        connections: The number of current connections to this instance.
    """

    robot_instance: NonNegativeInt
    # stream_id to list of tracks
    tracks: dict[str, list[RobotStreamTrack]]
    connections: int


class AvailableRobot(BaseModel):
    """Represents an available robot, including all its running instances.

    Attributes:
        robot_id: The unique identifier for the robot model/type.
        instances: A dictionary of all available instances for this robot,
                   keyed by instance ID.
    """

    robot_id: str
    instances: dict[int, AvailableRobotInstance]


class AvailableRobotCapacityUpdate(BaseModel):
    """Represents an update on the available capacity of all robots.

    This model is used to broadcast the current state of all available
    robots and their instances.

    Attributes:
        robots: A list of all available robots and their instances.
    """

    robots: list[AvailableRobot]


class BaseRecodingUpdatePayload(BaseModel):
    """Base payload for recording update notifications.

    Contains the minimum information needed to identify a recording
    and the robot instance it belongs to.
    """

    recording_id: str
    robot_id: str
    instance: NonNegativeInt


class RecodingRequestedPayload(BaseRecodingUpdatePayload):
    """Payload for recording request notifications.

    Contains information about who requested the recording and what
    data types should be captured.
    """

    created_by: str
    dataset_ids: list[str] = Field(default_factory=list)
    data_types: set[DataType] = Field(default_factory=set)


class RecordingStartPayload(RecodingRequestedPayload):
    """Payload for recording start notifications.

    Extends the request payload with the actual start timestamp
    when recording begins.
    """

    start_time: float


class RecordingNotificationType(str, Enum):
    """Types of recording lifecycle notifications."""

    INIT = "INIT"
    REQUESTED = "REQUESTED"
    START = "START"
    STOP = "STOP"
    SAVED = "SAVED"
    DISCARDED = "DISCARDED"
    EXPIRED = "EXPIRED"


class RecordingNotification(BaseModel):
    """Notification message for recording lifecycle events.

    Used to communicate recording state changes across the system,
    including initialization, start/stop events, and final disposition.
    """

    type: RecordingNotificationType
    payload: Union[
        RecordingStartPayload,
        RecodingRequestedPayload,
        list[Union[RecordingStartPayload, RecodingRequestedPayload]],
        BaseRecodingUpdatePayload,
    ]
    id: str = Field(default_factory=lambda: uuid4().hex)
