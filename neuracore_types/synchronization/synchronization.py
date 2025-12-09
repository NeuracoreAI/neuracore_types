"""Request models for dataset and recording synchronization operations."""

from pydantic import BaseModel, ConfigDict

from neuracore_types.episode.episode import RobotDataSpec


class SynchronizationDetails(BaseModel):
    """Details for synchronization requests.

    Attributes:
        frequency: Synchronization frequency in Hz.
        robot_data_spec: Specification of robot data to include in the synchronization.
        max_delay_s: Maximum allowable delay (in seconds) for synchronization.
        allow_duplicates: Whether to allow duplicate data points in the synchronization.
    """

    frequency: int
    robot_data_spec: RobotDataSpec | None
    max_delay_s: float = 0.1
    allow_duplicates: bool = True

    model_config = ConfigDict(frozen=True)

    def __hash__(self) -> int:
        """Compute a hash value for the SynchronizationDetails instance.

        Returns:
            int: The computed hash value.
        """
        # Convert the nested dict structure to something hashable
        robot_data_spec_hashable = None
        if self.robot_data_spec is not None:
            # Convert dict[str, dict[DataType, list[str]]] to a frozen structure
            robot_data_spec_hashable = tuple(
                sorted(
                    (
                        robot_name,
                        tuple(
                            sorted(
                                (data_type, tuple(fields))
                                for data_type, fields in data_spec.items()
                            )
                        ),
                    )
                    for robot_name, data_spec in self.robot_data_spec.items()
                )
            )

        return hash((
            self.frequency,
            robot_data_spec_hashable,
            self.max_delay_s,
            self.allow_duplicates,
        ))
