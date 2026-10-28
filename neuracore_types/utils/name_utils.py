"""Simple utility for converting names between user format and storage-safe format."""

import re
from urllib.parse import quote, unquote


def to_safe_name(name: str) -> str:
    """Convert user name to storage-safe name by URL encoding it, as ROS topics are.

    Args:
        name: The user-provided name.

    Returns:
        str: The storage-safe name.
    """
    return quote(name, safe="").replace(".", "%2E")


def from_safe_name(name: str) -> str:
    """Convert storage-safe name, URL-encoded or from older SDK versions, to user name.

    Args:
        name: The storage-safe name.

    Returns:
        str: The user-provided name.
    """
    return unquote(name.replace("\\", "/"))


def validate_safe_name(name: str) -> str:
    """Validate that a name is storage-safe.

    Args:
        name: The name to validate.

    Returns:
        str: The validated name.

    Raises:
        ValueError: If the name is not valid.
    """
    pattern = r"^[a-zA-Z0-9\-/._]{1,128}$"
    if not re.fullmatch(pattern, name):
        raise ValueError(
            f"Invalid name: {name}. Names must be alphanumeric and may "
            f"include '_', '-', '/', or '.'. Max length is 128 characters."
        )
    return to_safe_name(name)
