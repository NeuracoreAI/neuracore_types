"""Models for upload and WebRTC signaling."""

from datetime import datetime, timezone
from enum import Enum
from typing import NamedTuple, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt

from neuracore_types.nc_data import DataType
from neuracore_types.utils.pydantic_to_ts import (
    REQUIRED_WITH_DEFAULT_FLAG,
    fix_required_with_defaults,
)


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
    id: str = Field(
        default_factory=lambda: uuid4().hex,
        json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG,
    )

    model_config = ConfigDict(json_schema_extra=fix_required_with_defaults)


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
    id: str = Field(
        default_factory=lambda: uuid4().hex,
        json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG,
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG,
    )

    model_config = ConfigDict(json_schema_extra=fix_required_with_defaults)


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
    id: str = Field(
        default_factory=lambda: uuid4().hex,
        json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG,
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG,
    )

    model_config = ConfigDict(json_schema_extra=fix_required_with_defaults)


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


class RecordingStartPayload(BaseRecodingUpdatePayload):
    """Payload for recording start notifications."""

    created_by: str
    dataset_ids: list[str] = Field(
        default_factory=list, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    data_types: set[DataType] = Field(
        default_factory=set, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    start_time: float

    model_config = ConfigDict(json_schema_extra=fix_required_with_defaults)


class RecordingNotificationType(str, Enum):
    """Types of recording lifecycle notifications."""

    INIT = "INIT"
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
        list[RecordingStartPayload],
        BaseRecodingUpdatePayload,
    ]
    id: str = Field(
        default_factory=lambda: uuid4().hex,
        json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG,
    )

    model_config = ConfigDict(json_schema_extra=fix_required_with_defaults)


class RecordingDataTraceStatus(str, Enum):
    """Status for a recording data trace upload lifecycle."""

    QUEUED = "QUEUED"
    UPLOAD_STARTED = "UPLOAD_STARTED"
    UPLOAD_COMPLETE = "UPLOAD_COMPLETE"


class RecordingDataTrace(BaseModel):
    """Represents a single data trace belonging to a recording.

    This is used to track upload completion for each trace so that
    a recording can be saved once all traces are uploaded.
    """

    id: str
    recording_id: str
    data_type: DataType
    status: RecordingDataTraceStatus = Field(
        default=RecordingDataTraceStatus.QUEUED,
        json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG,
    )
    created_at: float = Field(
        default_factory=lambda: datetime.now().timestamp(),
        json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG,
    )
    uploaded_at: Optional[float]
    uploaded_bytes: int = Field(
        default=0,
        json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG,
    )
    total_bytes: int = Field(
        default=0,
        json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG,
    )

    model_config = ConfigDict(json_schema_extra=fix_required_with_defaults)


class RegisterTracesRequest(BaseModel):
    """Request to register traces for a recording from the data daemon.

    This is called when the data daemon stops recording. It provides
    all trace IDs that belong to the recording. Uploads can happen
    before or after this API call.
    """

    recording_id: str
    start_time: float
    end_time: float
    robot_id: Optional[str] = None
    robot_name: Optional[str] = None
    instance: NonNegativeInt
    dataset_id: Optional[str] = None
    dataset_name: Optional[str] = None
    is_shared: bool = Field(default=False, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG)
    traces: dict[str, int]  # trace_id -> total_bytes

    model_config = ConfigDict(json_schema_extra=fix_required_with_defaults)
