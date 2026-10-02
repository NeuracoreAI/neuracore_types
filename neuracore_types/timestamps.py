"""Integer timestamps in microseconds and their relation to float seconds.

A field named with the `_us` suffix holds an integer count of microseconds.
Microsecond values stay below 2**53, so they are exact as JavaScript numbers.
"""

import math
import time
from collections.abc import Callable
from typing import Annotated, Any, TypeVar

from pydantic import BaseModel, Field, StrictInt

MICROSECONDS_PER_SECOND = 1_000_000
NANOSECONDS_PER_MICROSECOND = 1_000
TIMESTAMP_US_LIMIT = 2**53

TimestampUs = Annotated[StrictInt, Field(ge=0, lt=TIMESTAMP_US_LIMIT)]

ModelT = TypeVar("ModelT", bound=BaseModel)


def now_us() -> int:
    """Return the wall clock in microseconds since the epoch."""
    return time.time_ns() // NANOSECONDS_PER_MICROSECOND


def seconds_to_us(seconds: float) -> int:
    """Round seconds to the nearest microsecond.

    Raises:
        ValueError: If the value in microseconds is not finite.
    """
    microseconds = seconds * MICROSECONDS_PER_SECOND
    if not math.isfinite(microseconds):
        raise ValueError(f"Seconds value {seconds} is not a finite microsecond value")
    return round(microseconds)


def add_microsecond_timestamps(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add `timestamp_us` to each trace entry that does not have it.

    The value is rounded from the entry's `timestamp` in seconds. When rounding
    does not advance past the previous converted entry, the value is the
    previous value plus one, so the converted entries stay strictly increasing.
    An entry that has `timestamp_us` is returned unchanged. Every other key is
    kept.
    """
    result = []
    previous_us: int | None = None
    for entry in entries:
        if "timestamp_us" not in entry:
            timestamp_us = seconds_to_us(entry["timestamp"])
            if previous_us is not None:
                timestamp_us = max(timestamp_us, previous_us + 1)
            entry = {**entry, "timestamp_us": timestamp_us}
            previous_us = timestamp_us
        result.append(entry)
    return result


def microseconds_field_from(seconds_field: str) -> Any:
    """Return a microsecond field that defaults to a rounded seconds field.

    The default is validated like an input value, so seconds outside the
    microsecond bounds are a validation error on the microsecond field. A
    value too large to convert defaults to TIMESTAMP_US_LIMIT, which the
    bound rejects. When the seconds field is absent, which only
    model_construct allows, the default is 0.

    Args:
        seconds_field: Name of the float seconds field, declared before the
            microsecond field.
    """

    def factory(validated_data: dict[str, Any]) -> int:
        if seconds_field not in validated_data:
            return 0
        try:
            return seconds_to_us(validated_data[seconds_field])
        except ValueError:
            return TIMESTAMP_US_LIMIT

    return Field(default_factory=factory, validate_default=True)


def seconds_from_microseconds(
    *field_pairs: tuple[str, str],
) -> Callable[[ModelT], ModelT]:
    """Return an after validator that sets each seconds field from microseconds.

    The microsecond field is the source of truth when both fields are given.

    Args:
        field_pairs: Pairs of (seconds field, microsecond field).
    """

    def set_seconds(model: ModelT) -> ModelT:
        for seconds_field, microseconds_field in field_pairs:
            microseconds = getattr(model, microseconds_field)
            # Not setattr, which would add the field to model_fields_set.
            model.__dict__[seconds_field] = microseconds / MICROSECONDS_PER_SECOND
        return model

    return set_seconds
