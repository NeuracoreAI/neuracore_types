"""Integer timestamps on a microsecond time base.

A tick is one microsecond. A float timestamp is legacy seconds and is
converted to ticks. Microseconds since the epoch stay below 2**53, so a tick
is exact as a JavaScript number.
"""

import math
import numbers
import time
from typing import Annotated, Any

from pydantic import BeforeValidator

TICKS_PER_SECOND = 1_000_000
NANOSECONDS_PER_TICK = 1_000
TICKS_LIMIT = 2**53


def now_ticks() -> int:
    """Return the monotonic clock in ticks."""
    return time.monotonic_ns() // NANOSECONDS_PER_TICK


def seconds_to_ticks(seconds: float) -> int:
    """Convert seconds to the nearest tick.

    Raises:
        ValueError: If the value in ticks is not finite.
    """
    ticks = seconds * TICKS_PER_SECOND
    if not math.isfinite(ticks):
        raise ValueError(f"Seconds value {seconds} is not a finite number of ticks")
    return round(ticks)


def timestamp_to_ticks(value: Any) -> int:
    """Convert a timestamp to ticks. An integer is ticks, a float is seconds.

    Raises:
        TypeError: If the value is a bool or not a number.
        ValueError: If the value is not finite or is at or above 2**53 ticks.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"Timestamp must be an int or a float, got {type(value)}")
    if isinstance(value, numbers.Integral):
        ticks = int(value)
    else:
        ticks = seconds_to_ticks(float(value))
    if ticks >= TICKS_LIMIT:
        raise ValueError(f"Timestamp {ticks} is at or above 2**53 ticks")
    return ticks


def convert_legacy_trace_entries(entries: list[dict]) -> list[dict]:
    """Convert float second timestamps in trace entries to strictly increasing ticks.

    A float timestamp becomes one tick after the previous entry when rounding
    would not advance it. Integer timestamps are kept, so running the
    conversion on its own output changes nothing.
    """
    converted = []
    previous_ticks: int | None = None
    for entry in entries:
        timestamp = entry["timestamp"]
        if isinstance(timestamp, float):
            ticks = seconds_to_ticks(timestamp)
            if previous_ticks is not None:
                ticks = max(ticks, previous_ticks + 1)
            entry = {**entry, "timestamp": ticks}
        converted.append(entry)
        previous_ticks = entry["timestamp"]
    return converted


def _validate_ticks(value: Any) -> int:
    try:
        return timestamp_to_ticks(value)
    except TypeError as error:
        raise ValueError(str(error)) from error


Ticks = Annotated[int, BeforeValidator(_validate_ticks)]
