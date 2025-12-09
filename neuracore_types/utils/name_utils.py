"""Simple utility for converting names between user format and storage-safe format."""

import re


def to_safe_name(name: str) -> str:
    r"""Convert user name to storage-safe name by replacing / with \\.

    Args:
        name: The user-provided name.

    Returns:
        str: The storage-safe name.
    """
    return name.replace("/", "\\")


def from_safe_name(name: str) -> str:
    r"""Convert storage-safe name back to user name by replacing \\ with /.

    Args:
        name: The storage-safe name.

    Returns:
        str: The user-provided name.
    """
    return name.replace("\\", "/")


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
    if not re.match(pattern, name):
        raise ValueError(
            f"Invalid name: {name}. Names must be alphanumeric and may "
            f"include '_', '-', '/', or '.'. Max length is 128 characters."
        )
    return to_safe_name(name)
