"""Base classes for Neuracore data types."""

from typing import Any, Callable, Union

import torch
from pydantic import BaseModel, ConfigDict

from neuracore_types.nc_data.nc_data import NCData


class BatchedNCData(BaseModel):
    """Base class for batched Neuracore data."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to(self, device: torch.device) -> "BatchedNCData":
        """Move all tensors to the specified device."""
        data_dict = self.model_dump()
        moved_data = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in data_dict.items()
        }
        return self.__class__(**moved_data)

    @classmethod
    def from_nc_data(cls, nc_data: NCData) -> "BatchedNCData":
        """Create BatchedNCData from NCData by adding time and batch dimensions."""
        raise NotImplementedError(
            "from_nc_data method must be implemented in subclasses."
        )

    @classmethod
    def from_nc_data_list(cls, nc_data_list: list[NCData]) -> "BatchedNCData":
        """Create BatchedNCData from list of NCData, stacking along time dimension.

        This method is more efficient than calling from_nc_data() multiple times
        and concatenating, as it creates a single tensor directly.

        Args:
            nc_data_list: List of NCData instances to convert

        Returns:
            BatchedNCData with shape (1, T, ...) where T = len(nc_data_list)
        """
        raise NotImplementedError(
            "from_nc_data_list method must be implemented in subclasses."
        )

    @classmethod
    def sample(cls, batch_size: int = 1, time_steps: int = 1) -> "BatchedNCData":
        """Sample an example instance of BatchedNCData."""
        raise NotImplementedError("sample method must be implemented in subclasses.")

    @staticmethod
    def _create_tensor_handlers(
        field_name: str,
    ) -> tuple[
        Callable[[Union[dict[str, Any], torch.Tensor]], torch.Tensor],
        Callable[[torch.Tensor], dict[str, Any]],
    ]:
        """Create validator and serializer for a torch.Tensor field."""

        def validator(v: Union[dict[str, Any], torch.Tensor]) -> torch.Tensor:
            if isinstance(v, torch.Tensor):
                return v
            elif isinstance(v, dict) and v.get("_tensor_encoded"):
                # Convert list back to tensor with proper dtype and shape
                dtype_str = v["dtype"]
                # Map numpy dtype strings to torch dtypes
                dtype_map = {
                    "float32": torch.float32,
                    "float64": torch.float64,
                    "float16": torch.float16,
                    "int32": torch.int32,
                    "int64": torch.int64,
                    "int16": torch.int16,
                    "int8": torch.int8,
                    "uint8": torch.uint8,
                    "bool": torch.bool,
                }
                torch_dtype = dtype_map.get(dtype_str, torch.float32)
                return torch.tensor(v["data"], dtype=torch_dtype).reshape(v["shape"])
            else:
                raise ValueError(f"Invalid value for {field_name}")

        def serializer(v: torch.Tensor) -> dict[str, Any]:
            return {
                "_tensor_encoded": True,
                "shape": list(v.shape),
                "dtype": str(v.dtype).replace("torch.", ""),  # e.g., "float32"
                "data": v.cpu().numpy().tolist(),
            }

        return validator, serializer
