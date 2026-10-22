"""Request and response models for data synthesis."""

from pydantic import BaseModel, Field

from neuracore_types.data_synthesis.data_synthesis import (
    DataSynthesisAlgorithmConfig,
    DataSynthesisStage,
)


class DataSynthesisJobRequest(BaseModel):
    """Request model for starting a data synthesis job.

    Attributes:
        name: Name for the synthesis job.
        dataset_id: Identifier of the dataset to expand.
        output_dataset_name: Name of the dataset the job writes to.
        algorithm: Configuration of the algorithm to run.
        scaling_factor: How many recordings to produce per original. At
            least one, since a job that produces nothing has no purpose.
        include_original: Whether the output dataset should also contain the
            source recordings. They are referenced, never copied.
    """

    name: str
    dataset_id: str
    output_dataset_name: str
    algorithm: DataSynthesisAlgorithmConfig
    scaling_factor: int = Field(ge=1, le=10)
    include_original: bool = False


class InternalStartDataSynthesisJobRequest(BaseModel):
    """Task payload for provisioning a data synthesis job.

    Attributes:
        org_id: The ID of the organization.
        user_id: The ID of the user who started the job.
        user_name: The display name of that user, recorded on the datasets and
            recordings the job creates.
        job_uuid: The UUID of the job.
        dataset_id: The ID of the dataset to expand.
        job_name: The name of the job.
        output_dataset_name: The name of the dataset the job writes to.
        algorithm: Configuration of the algorithm to run.
        scaling_factor: How many recordings to produce per original.
        include_original: Whether the output dataset should also reference the
            source recordings.
    """

    org_id: str
    user_id: str
    user_name: str
    job_uuid: str
    dataset_id: str
    job_name: str
    output_dataset_name: str
    algorithm: DataSynthesisAlgorithmConfig
    scaling_factor: int
    include_original: bool = False


class DataSynthesisJobUpdateRequest(BaseModel):
    """Progress reported by a running synthesis job.

    Attributes:
        stage: The pipeline step the job is on.
        recordings_total: How many recordings the job expects to synthesize.
        recordings_completed: How many recordings the job has synthesized.
        error: An error message, if the job has failed.
    """

    stage: DataSynthesisStage | None = None
    recordings_total: int | None = None
    recordings_completed: int | None = None
    error: str | None = None


class SyntheticRecordingCreateRequest(BaseModel):
    """Request to stage a new recording derived from an existing one.

    Every trace of the source recording other than its RGB video is copied
    server side, so the caller only has to upload the regenerated video.

    Attributes:
        job_id: The ID of the synthesis job the recording belongs to.
        source_recording_id: The recording the new recording is derived from.
    """

    job_id: str
    source_recording_id: str


class SyntheticRecordingCreateResponse(BaseModel):
    """Where to upload the regenerated video for a staged recording.

    Attributes:
        recording_id: The ID minted for the new recording.
        video_upload_urls: Resumable upload URI per RGB sensor name.
    """

    recording_id: str
    video_upload_urls: dict[str, str]


class SyntheticRecordingFinalizeRequest(BaseModel):
    """Request to publish a staged recording into the output dataset.

    Attributes:
        job_id: The ID of the synthesis job the recording belongs to.
        source_recording_id: The recording the new recording was derived from.
        dataset_id: The dataset the recording is added to.
    """

    job_id: str
    source_recording_id: str
    dataset_id: str


class DataSynthesisPreviewRequest(BaseModel):
    """Request example frames showing what a job would generate.

    Attributes:
        dataset_id: Identifier of the dataset to sample from.
        algorithm: Configuration of the algorithm to run on those frames.
        recording_index: Which recording of the dataset to sample from.
        frame_index: Which frame of each camera's video to sample.
    """

    dataset_id: str
    algorithm: DataSynthesisAlgorithmConfig
    recording_index: int = 0
    frame_index: int = 0


class DataSynthesisPreviewFrame(BaseModel):
    """One camera's frame, alongside what synthesis made of it.

    Attributes:
        camera: The RGB sensor the frames came from.
        before_url: Signed URL of the original frame.
        after_url: Signed URL of the generated frame.
    """

    camera: str
    before_url: str
    after_url: str


class DataSynthesisPreviewResponse(BaseModel):
    """Generated example frames alongside the frames they came from.

    Every camera is shown, because a prompt that segments the right object
    from one viewpoint can easily miss from another, and a job regenerates
    all of them.

    Attributes:
        frames: One entry per camera the recording holds, in the order the
            recording lists them.
        config_hash: Digest of the configuration that produced this preview,
            used to tell whether a preview still matches the current settings.
    """

    frames: list[DataSynthesisPreviewFrame]
    config_hash: str
