"""Data models for natural language data."""

import os
from typing import Any, Literal, cast

import torch
from pydantic import field_serializer, field_validator

from neuracore_types.batched_nc_data.batched_nc_data import BatchedNCData
from neuracore_types.nc_data.language_data import LanguageData
from neuracore_types.nc_data.nc_data import NCData

LANGUAGE_MODEL_NAME = os.getenv("LANGUAGE_MODEL_NAME", "distilbert-base-uncased")

_tokenizer = None


class BatchedLanguageData(BatchedNCData):
    """Batched natural language data for sequences of text inputs."""

    type: Literal["BatchedLanguageData"] = "BatchedLanguageData"
    input_ids: torch.Tensor  # (B, T, L) int64
    attention_mask: torch.Tensor  # (B, T, L) float32

    @field_validator("input_ids", mode="before")
    @classmethod
    def decode_input_ids(cls, v: dict[str, Any]) -> torch.Tensor:
        """Decode input_ids field to torch.Tensor."""
        return cls._create_tensor_handlers("input_ids")[0](v)

    @field_serializer("input_ids", when_used="json")
    def serialize_input_ids(self, v: torch.Tensor) -> dict[str, Any]:
        """Serialize input_ids field to base64 string."""
        return self._create_tensor_handlers("input_ids")[1](v)

    @field_validator("attention_mask", mode="before")
    @classmethod
    def decode_attention_mask(cls, v: dict[str, Any]) -> torch.Tensor:
        """Decode attention_mask field to torch.Tensor."""
        return cls._create_tensor_handlers("attention_mask")[0](v)

    @field_serializer("attention_mask", when_used="json")
    def serialize_attention_mask(self, v: torch.Tensor) -> dict[str, Any]:
        """Serialize attention_mask field to base64 string."""
        return self._create_tensor_handlers("attention_mask")[1](v)

    @classmethod
    def _tokenize(cls, text: str) -> dict:
        """Tokenize input text using the tokenizer.

        Args:
            text: Input text string

        Returns:
            dict: Tokenized output containing input_ids and attention_mask
        """
        global _tokenizer
        if _tokenizer is None:
            from transformers import AutoTokenizer

            _tokenizer = AutoTokenizer.from_pretrained(LANGUAGE_MODEL_NAME)
        tokens = _tokenizer(
            text,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return tokens

    @classmethod
    def from_nc_data(cls, nc_data: NCData) -> "BatchedNCData":
        """Create BatchedLanguageData from LanguageData.

        Args:
            nc_data: NCData instance to convert

        Returns:
            BatchedNCData: Converted BatchedNCData instance
        """
        language_data: LanguageData = cast(LanguageData, nc_data)
        tokens = cls._tokenize(language_data.text)
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        # Add T dimension
        return cls(
            input_ids=input_ids.unsqueeze(1),  # (1, L) -> (1, 1, L)
            attention_mask=attention_mask.unsqueeze(1),  # (1, L) -> (1, 1, L)
        )

    @classmethod
    def sample(cls, batch_size: int = 1, time_steps: int = 1) -> "BatchedLanguageData":
        """Sample an example instance of BatchedLanguageData.

        Args:
            batch_size: Number of samples in the batch
            time_steps: Number of time steps in the sequence

        Returns:
            BatchedLanguageData: Sampled BatchedLanguageData instance
        """
        tokens = cls._tokenize(LanguageData.sample().text)
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        return cls(
            input_ids=input_ids.unsqueeze(1).repeat(
                batch_size, time_steps, 1
            ),  # (1, L) -> (B, T, L)
            attention_mask=attention_mask.unsqueeze(1).repeat(
                batch_size, time_steps, 1
            ),  # (1, L) -> (B, T, L)
        )
