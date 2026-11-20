"""Model constructor arguments sent with a training request."""

from pydantic import BaseModel, ConfigDict, model_validator

from neuracore_types.synchronization.synchronization import SynchronizationConfig
from neuracore_types.training.training_config import TrainingConfig


class AlgorithmConfig(BaseModel):
    """Constructor arguments of the selected algorithm.

    Every key passes through to the algorithm unchanged. The model rejects
    keys that name a training or synchronization setting, and keys that start
    with an underscore, such as the Hydra key _target_.
    """

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def reject_misplaced_keys(self) -> "AlgorithmConfig":
        """Raise on training, synchronization and underscore keys."""
        algorithm_keys = set(self.model_extra or {})
        misplaced_training_keys = algorithm_keys & set(TrainingConfig.model_fields)
        misplaced_synchronization_keys = algorithm_keys & set(
            SynchronizationConfig.model_fields
        )
        misplaced_key_messages = []
        underscore_keys = {key for key in algorithm_keys if key.startswith("_")}
        if underscore_keys:
            misplaced_key_messages.append(
                f"{', '.join(sorted(underscore_keys))} are not algorithm arguments"
            )
        if misplaced_training_keys:
            misplaced_key_messages.append(
                f"{', '.join(sorted(misplaced_training_keys))} belong in"
                " training_config"
            )
        if misplaced_synchronization_keys:
            misplaced_key_messages.append(
                f"{', '.join(sorted(misplaced_synchronization_keys))} belong in"
                " synchronization_config"
            )
        if misplaced_key_messages:
            raise ValueError("; ".join(misplaced_key_messages))
        return self
