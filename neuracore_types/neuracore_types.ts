/* tslint:disable */
/* eslint-disable */
/**
/* This file was automatically generated from pydantic models by running pydantic2ts.
/* Do not modify it by hand - just update the pydantic models and then re-run the script
*/

/**
 * Represents an available robot, including all its running instances.
 *
 * Attributes:
 *     robot_id: The unique identifier for the robot model/type.
 *     instances: A dictionary of all available instances for this robot,
 *                keyed by instance ID.
 */
export interface AvailableRobot {
  robot_id: string;
  instances: {
    [k: string]: AvailableRobotInstance;
  };
}
/**
 * Represents a single, available instance of a robot.
 *
 * Attributes:
 *     robot_instance: The unique identifier for this robot instance.
 *     tracks: A dictionary of available media stream tracks for this instance.
 *     connections: The number of current connections to this instance.
 */
export interface AvailableRobotInstance {
  robot_instance: number;
  tracks: {
    [k: string]: RobotStreamTrack[];
  };
  connections: number;
}
/**
 * Metadata for a robot's media stream track.
 *
 * This model holds all the necessary information to identify and manage
 * a single media track (e.g., a video or audio feed) from a specific
 * robot instance.
 *
 * Attributes:
 *     robot_id: The unique identifier of the robot providing the stream.
 *     robot_instance: The specific instance number of the robot.
 *     stream_id: The identifier for the overall media stream session.
 *     kind: The type of media track, typically 'audio' or 'video'.
 *     label: A human-readable label for the track (e.g., 'front_camera').
 *     mid: The media ID used in SDP, essential for WebRTC negotiation.
 *     id: A unique identifier for this track metadata object.
 *     created_at: The UTC timestamp when this track metadata was created.
 */
export interface RobotStreamTrack {
  robot_id: string;
  robot_instance: number;
  stream_id: string;
  kind: TrackKind;
  label: string;
  mid: string;
  id?: string;
  created_at?: string;
}
/**
 * Represents an update on the available capacity of all robots.
 *
 * This model is used to broadcast the current state of all available
 * robots and their instances.
 *
 * Attributes:
 *     robots: A list of all available robots and their instances.
 */
export interface AvailableRobotCapacityUpdate {
  robots: AvailableRobot[];
}
/**
 * Base payload for recording update notifications.
 *
 * Contains the minimum information needed to identify a recording
 * and the robot instance it belongs to.
 */
export interface BaseRecodingUpdatePayload {
  recording_id: string;
  robot_id: string;
  instance: number;
}
/**
 * Camera sensor data including images and calibration information.
 *
 * Contains image data along with camera intrinsic and extrinsic parameters
 * for 3D reconstruction and computer vision applications. The frame field
 * is populated during dataset iteration for efficiency.
 */
export interface CameraData {
  timestamp?: number;
  frame_idx?: number;
  extrinsics?: number[][] | null;
  intrinsics?: number[][] | null;
  frame?: unknown;
}
/**
 * Generic container for application-specific data types.
 *
 * Provides a flexible way to include custom sensor data or application-specific
 * information that doesn't fit into the standard data categories.
 */
export interface CustomData {
  timestamp?: number;
  data: unknown;
}
/**
 * Statistical summary of data dimensions and distributions.
 *
 * Contains statistical information about data arrays including means,
 * standard deviations, counts, and maximum lengths for normalization
 * and model configuration purposes.
 *
 * Attributes:
 *     mean: List of means for each data dimension
 *     std: List of standard deviations for each data dimension
 *     count: List of counts for each data dimension
 *     min: List of minimum values for each data dimension
 *     max: List of maximum values for each data dimension
 *     max_len: Maximum length of the data arrays
 *     robot_to_ncdata_keys: Mapping of robot ids to their associated
 *         data keys for this data type
 */
export interface DataItemStats {
  mean?: number[];
  std?: number[];
  count?: number[];
  min?: number[];
  max?: number[];
  max_len?: number;
  robot_to_ncdata_keys?: {
    [k: string]: string[];
  };
}
/**
 * Represents a dataset of robot demonstrations.
 *
 * A dataset groups related robot demonstrations together and maintains metadata
 * about the collection as a whole.
 *
 * Attributes:
 *     id: Unique identifier for the dataset.
 *     name: Human-readable name for the dataset.
 *     created_at: Unix timestamp of dataset creation.
 *     modified_at: Unix timestamp of last modification.
 *     description: Optional description of the dataset.
 *     tags: List of tags for categorizing the dataset.
 *     recording_ids: List of recording IDs in this dataset
 *     demonstration_ids: List of demonstration IDs in this dataset.
 *     num_demonstrations: Total number of demonstrations.
 *     total_duration_seconds: Total duration of all demonstrations.
 *     size_bytes: Total size of all demonstrations.
 *     is_shared: Whether the dataset is shared with other users.
 *     metadata: Additional arbitrary metadata.
 *     synced_dataset_ids: List of synced dataset IDs in this dataset.
 *                         They point to synced datasets that synchronized
 *                         this dataset at a particular frequency.
 */
export interface Dataset {
  id: string;
  name: string;
  created_at: number;
  modified_at: number;
  description?: string | null;
  tags?: string[];
  recording_ids?: string[];
  num_demonstrations?: number;
  total_duration_seconds?: number;
  size_bytes?: number;
  is_shared?: boolean;
  metadata?: {
    [k: string]: unknown;
  };
  synced_dataset_ids?: {
    [k: string]: unknown;
  };
  all_data_types?: {
    [k: string]: number;
  };
  common_data_types?: {
    [k: string]: number;
  };
  recording_ids_in_bucket?: boolean;
}
/**
 * Comprehensive description of dataset contents and statistics.
 *
 * Provides metadata about a complete dataset including statistical summaries
 * for all data types, maximum counts for variable-length data, and methods
 * for determining which data types are present.
 */
export interface DatasetDescription {
  total_num_transitions?: number;
  joint_positions?: DataItemStats;
  joint_velocities?: DataItemStats;
  joint_torques?: DataItemStats;
  joint_target_positions?: DataItemStats;
  end_effector_states?: DataItemStats;
  end_effector_poses?: DataItemStats;
  parallel_gripper_open_amounts?: DataItemStats;
  poses?: DataItemStats;
  rgb_images?: DataItemStats;
  depth_images?: DataItemStats;
  point_clouds?: DataItemStats;
  language?: DataItemStats;
  custom_data?: {
    [k: string]: DataItemStats;
  };
}
/**
 * End-effector state data including gripper and tool configurations.
 *
 * Contains the state of robot end-effectors such as gripper opening amounts,
 * tool activations, or other end-effector specific parameters.
 */
export interface EndEffectorData {
  timestamp?: number;
  open_amounts: {
    [k: string]: number;
  };
}
/**
 * End-effector pose data.
 *
 * Contains the pose of end-effectors as a 7-element list containing the
 * position and unit quaternion orientation [x, y, z, qx, qy, qz, qw].
 */
export interface EndEffectorPoseData {
  timestamp?: number;
  poses: {
    [k: string]: number[];
  };
}
/**
 * Represents a signaling message for the WebRTC handshake process.
 *
 * This message is exchanged between two peers via a signaling server to
 * negotiate the connection details, such as SDP offers/answers and ICE
 * candidates.
 *
 * Attributes:
 *     from_id: The unique identifier of the sender peer.
 *     to_id: The unique identifier of the recipient peer.
 *     data: The payload of the message, typically an SDP string or a JSON
 *           object with ICE candidate information.
 *     connection_id: The unique identifier for the connection session.
 *     type: The type of the handshake message, as defined by MessageType.
 *     id: A unique identifier for the message itself.
 */
export interface HandshakeMessage {
  from_id: string;
  to_id: string;
  data: string;
  connection_id: string;
  type: MessageType;
  id?: string;
}
/**
 * Robot joint state data including positions, velocities, or torques.
 *
 * Represents joint-space data for robotic systems with support for named
 * joints and additional auxiliary values. Used for positions, velocities,
 * torques, and target positions.
 */
export interface JointData {
  timestamp?: number;
  values: {
    [k: string]: number;
  };
  additional_values?: {
    [k: string]: number;
  } | null;
}
/**
 * Natural language instruction or description data.
 *
 * Contains text-based information such as task descriptions, voice commands,
 * or other linguistic data associated with robot demonstrations.
 */
export interface LanguageData {
  timestamp?: number;
  text: string;
}
/**
 * Configuration specification for initializing Neuracore models.
 *
 * Defines the model architecture requirements including dataset characteristics,
 * input/output data types, and prediction horizons for model initialization
 * and training configuration.
 */
export interface ModelInitDescription {
  dataset_description: DatasetDescription;
  input_data_types: DataType[];
  output_data_types: DataType[];
  output_prediction_horizon?: number;
}
/**
 * Model inference output containing predictions and timing information.
 *
 * Represents the results of model inference including predicted outputs
 * for each configured data type and optional timing information for
 * performance monitoring.
 */
export interface ModelPrediction {
  outputs?: {
    [k: string]: unknown;
  };
  prediction_time?: number | null;
}
/**
 * Base class for all Neuracore data with automatic timestamping.
 *
 * Provides a common base for all data types in the system with automatic
 * timestamp generation for temporal synchronization and data ordering.
 */
export interface NCData {
  timestamp?: number;
}
/**
 * The details describing properties about the new connection.
 *
 * Attributes:
 *     connection_token: The token used for security to establish the connection.
 *     robot_id: The unique identifier for the robot to connect to
 *     robot_instance: The identifier for the instance of the robot to connect to.
 *     video_format: The type of video the consumer expects to receive.
 */
export interface OpenConnectionDetails {
  connection_token: string;
  robot_id: string;
  robot_instance: number;
  video_format: VideoFormat;
}
/**
 * Represents a request to open a new WebRTC connection.
 *
 * Attributes:
 *     from_id: The unique identifier of the consumer peer.
 *     to_id: The unique identifier of the producer peer.
 *     robot_id: The unique identifier for the robot to be created.
 *     robot_instance: The identifier for the instance of the robot to connect to.
 *     video_format: The type of video the consumer expects to receive.
 *     id: the identifier for this connection request.
 *     created_at: when the request was created.
 */
export interface OpenConnectionRequest {
  from_id: string;
  to_id: string;
  robot_id: string;
  robot_instance: number;
  video_format: VideoFormat;
  id?: string;
  created_at?: string;
}
/**
 * Open amount data for parallel end effector gripper.
 *
 * Contains the state of parallel gripper opening amounts.
 */
export interface ParallelGripperOpenAmountData {
  timestamp?: number;
  open_amounts: {
    [k: string]: number;
  };
}
/**
 * 3D point cloud data with optional RGB colouring and camera parameters.
 *
 * Represents 3D spatial data from depth sensors or LiDAR systems with
 * optional colour information and camera calibration for registration.
 */
export interface PointCloudData {
  timestamp?: number;
  points?: string | null;
  rgb_points?: string | null;
  extrinsics?: string | null;
  intrinsics?: string | null;
}
/**
 * 6DOF pose data for objects, end-effectors, or coordinate frames.
 *
 * Represents position and orientation information for tracking objects
 * or robot components in 3D space. Poses are stored as dictionaries
 * mapping pose names to [x, y, z, rx, ry, rz] values.
 */
export interface PoseData {
  timestamp?: number;
  pose: {
    [k: string]: number[];
  };
}
/**
 * Payload for recording request notifications.
 *
 * Contains information about who requested the recording and what
 * data types should be captured.
 */
export interface RecodingRequestedPayload {
  recording_id: string;
  robot_id: string;
  instance: number;
  created_by: string;
  dataset_ids?: string[];
  data_types?: DataType[];
}
/**
 * Description of a single recording episode with statistics and counts.
 *
 * Provides metadata about an individual recording including data statistics,
 * sensor counts, and episode length for analysis and processing.
 */
export interface RecordingDescription {
  joint_positions?: DataItemStats;
  joint_velocities?: DataItemStats;
  joint_torques?: DataItemStats;
  joint_target_positions?: DataItemStats;
  end_effector_states?: DataItemStats;
  end_effector_poses?: DataItemStats;
  parallel_gripper_open_amounts?: DataItemStats;
  poses?: DataItemStats;
  rgb_images?: DataItemStats;
  depth_images?: DataItemStats;
  point_clouds?: DataItemStats;
  language?: DataItemStats;
  episode_length?: number;
  custom_data?: {
    [k: string]: DataItemStats;
  };
}
/**
 * Notification message for recording lifecycle events.
 *
 * Used to communicate recording state changes across the system,
 * including initialization, start/stop events, and final disposition.
 */
export interface RecordingNotification {
  type: RecordingNotificationType;
  payload:
    | RecordingStartPayload
    | RecodingRequestedPayload
    | (RecordingStartPayload | RecodingRequestedPayload)[]
    | BaseRecodingUpdatePayload;
  id?: string;
}
/**
 * Payload for recording start notifications.
 *
 * Extends the request payload with the actual start timestamp
 * when recording begins.
 */
export interface RecordingStartPayload {
  recording_id: string;
  robot_id: string;
  instance: number;
  created_by: string;
  dataset_ids?: string[];
  data_types?: DataType[];
  start_time: number;
}
/**
 * Represents the response from asserting a stream is alive.
 *
 * This is returned when a client pings a stream to keep it active.
 *
 * Attributes:
 *     resurrected: A boolean indicating if the stream was considered dead
 *                  and has been successfully resurrected by this request.
 */
export interface StreamAliveResponse {
  resurrected: boolean;
}
/**
 * Synchronized collection of all sensor data at a single time point.
 *
 * Represents a complete snapshot of robot state and sensor information
 * at a specific timestamp. Used for creating temporally aligned datasets
 * and ensuring consistent data relationships across different sensors.
 */
export interface SyncPoint {
  timestamp?: number;
  joint_positions?: JointData | null;
  joint_velocities?: JointData | null;
  joint_torques?: JointData | null;
  joint_target_positions?: JointData | null;
  end_effectors?: EndEffectorData | null;
  end_effector_poses?: EndEffectorPoseData | null;
  parallel_gripper_open_amounts?: ParallelGripperOpenAmountData | null;
  poses?: PoseData | null;
  rgb_images?: {
    [k: string]: CameraData;
  } | null;
  depth_images?: {
    [k: string]: CameraData;
  } | null;
  point_clouds?: {
    [k: string]: PointCloudData;
  } | null;
  language_data?: LanguageData | null;
  custom_data?: {
    [k: string]: CustomData;
  } | null;
  robot_id?: string | null;
}
/**
 * Complete synchronized dataset containing a sequence of data points.
 *
 * Represents an entire recording or demonstration as a time-ordered sequence
 * of synchronized data points with start and end timestamps for temporal
 * reference.
 */
export interface SyncedData {
  frames: SyncPoint[];
  start_time: number;
  end_time: number;
  robot_id: string;
}
/**
 * Represents a dataset of robot demonstrations.
 *
 * A Synchronized dataset groups related robot demonstrations together
 * and maintains metadata about the collection as a whole.
 *
 * Attributes:
 *     id: Unique identifier for the synced dataset.
 *     parent_id: Unique identifier of the corresponding dataset.
 *     freq: Frequency at which dataset was processed.
 *     name: Human-readable name for the dataset.
 *     created_at: Unix timestamp of dataset creation.
 *     modified_at: Unix timestamp of last modification.
 *     description: Optional description of the dataset.
 *     recording_ids: List of recording IDs in this dataset
 *     num_demonstrations: Total number of demonstrations.
 *     total_duration_seconds: Total duration of all demonstrations.
 *     is_shared: Whether the dataset is shared with other users.
 *     metadata: Additional arbitrary metadata.
 */
export interface SyncedDataset {
  id: string;
  parent_id: string;
  freq: number;
  name: string;
  created_at: number;
  modified_at: number;
  description?: string | null;
  recording_ids?: string[];
  num_demonstrations?: number;
  num_processed_demonstrations?: number;
  total_duration_seconds?: number;
  is_shared?: boolean;
  metadata?: {
    [k: string]: unknown;
  };
  dataset_description?: DatasetDescription;
  all_data_types?: {
    [k: string]: number;
  };
  common_data_types?: {
    [k: string]: number;
  };
}

/**
 * Enumerates the supported track kinds for streaming.
 */
export enum TrackKind {
  JOINTS = "JOINTS",
  RGB = "RGB",
  DEPTH = "DEPTH",
  LANGUAGE = "LANGUAGE",
  GRIPPER = "GRIPPER",
  END_EFFECTOR_POSE = "END_EFFECTOR_POSE",
  PARALLEL_GRIPPER_OPEN_AMOUNT = "PARALLEL_GRIPPER_OPEN_AMOUNT",
  POINT_CLOUD = "POINT_CLOUD",
  POSE = "POSE",
  CUSTOM = "CUSTOM"
}
/**
 * Enumerates the types of signaling messages for WebRTC handshakes.
 *
 * These types are used to identify the purpose of a message sent through
 * the signaling server during connection establishment.
 */
export enum MessageType {
  SDP_OFFER = "SDP_OFFER",
  SDP_ANSWER = "SDP_ANSWER",
  ICE_CANDIDATE = "ICE_CANDIDATE",
  OPEN_CONNECTION = "OPEN_CONNECTION"
}
/**
 * Enumeration of supported data types in the Neuracore system.
 *
 * Defines the standard data categories used for dataset organization,
 * model training, and data processing pipelines.
 */
export enum DataType {
  JOINT_POSITIONS = "JOINT_POSITIONS",
  JOINT_VELOCITIES = "JOINT_VELOCITIES",
  JOINT_TORQUES = "JOINT_TORQUES",
  JOINT_TARGET_POSITIONS = "JOINT_TARGET_POSITIONS",
  END_EFFECTORS = "END_EFFECTORS",
  END_EFFECTOR_POSES = "END_EFFECTOR_POSES",
  PARALLEL_GRIPPER_OPEN_AMOUNTS = "PARALLEL_GRIPPER_OPEN_AMOUNTS",
  RGB_IMAGE = "RGB_IMAGE",
  DEPTH_IMAGE = "DEPTH_IMAGE",
  POINT_CLOUD = "POINT_CLOUD",
  POSES = "POSES",
  LANGUAGE = "LANGUAGE",
  CUSTOM = "CUSTOM"
}
/**
 * Enumerates video format styles over a WebRTC connection.
 */
export enum VideoFormat {
  WEB_RTC_NEGOTIATED = "WEB_RTC_NEGOTIATED",
  NEURACORE_CUSTOM = "NEURACORE_CUSTOM"
}
/**
 * Types of recording lifecycle notifications.
 */
export enum RecordingNotificationType {
  INIT = "INIT",
  REQUESTED = "REQUESTED",
  START = "START",
  STOP = "STOP",
  SAVED = "SAVED",
  DISCARDED = "DISCARDED",
  EXPIRED = "EXPIRED"
}
