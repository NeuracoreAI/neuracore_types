"""Tests for name_converter.py"""

import pytest

from neuracore_types.utils import from_safe_name, to_safe_name, validate_safe_name


def test_basic_conversion():
    assert to_safe_name("simple") == "simple"
    assert from_safe_name("simple") == "simple"


def test_slash_conversion():
    assert to_safe_name("path/to/item") == "path\\to\\item"
    assert from_safe_name("path\\to\\item") == "path/to/item"


def test_roundtrip():
    original = "my/path/name"
    safe = to_safe_name(original)
    restored = from_safe_name(safe)
    assert original == restored


def test_validation():
    assert validate_safe_name("valid_name") == "valid_name"
    assert validate_safe_name("with-dash") == "with-dash"
    assert validate_safe_name("with.dot") == "with.dot"
    assert validate_safe_name("path/to/item") == "path\\to\\item"


def test_validation_errors():
    with pytest.raises(ValueError):
        validate_safe_name("has space")

    with pytest.raises(ValueError):
        validate_safe_name("has@symbol")
