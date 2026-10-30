"""Tests for name_converter.py"""

import random
import re
import string

import pytest

from neuracore_types.utils import from_safe_name, to_safe_name, validate_safe_name

VALID_CHARACTERS = string.ascii_letters + string.digits + "-/._"
ALL_CHARACTERS = [chr(i) for i in range(128)] + ["é", "日", "🤖"]


def _generated_names() -> list[str]:
    """Every valid name of one or two characters, plus random longer ones."""
    rng = random.Random(0)
    return [
        *VALID_CHARACTERS,
        *(a + b for a in VALID_CHARACTERS for b in VALID_CHARACTERS),
        *(
            "".join(rng.choices(VALID_CHARACTERS, k=rng.randint(3, 128)))
            for _ in range(1000)
        ),
    ]


GENERATED_NAMES = _generated_names()


def test_basic_conversion():
    assert to_safe_name("simple") == "simple"
    assert from_safe_name("simple") == "simple"


def test_slash_conversion():
    assert to_safe_name("path/to/item") == "path%2Fto%2Fitem"
    assert from_safe_name("path%2Fto%2Fitem") == "path/to/item"


def test_dot_conversion():
    assert to_safe_name("arm0.el0") == "arm0%2Eel0"
    assert to_safe_name("..") == "%2E%2E"
    assert from_safe_name("arm0%2Eel0") == "arm0.el0"


def test_matches_ros_topic_escaping():
    assert to_safe_name("camera_left/rgb.v2-a") == "camera_left%2Frgb%2Ev2-a"


def test_legacy_conversion():
    assert from_safe_name("path\\to\\item") == "path/to/item"
    assert from_safe_name("with.dot") == "with.dot"


def test_generated_names_escape_only_slashes_and_dots():
    for name in GENERATED_NAMES:
        safe = to_safe_name(name)
        assert safe == name.replace("/", "%2F").replace(".", "%2E"), name
        assert re.fullmatch(r"[a-zA-Z0-9\-_%]+", safe), name
        assert validate_safe_name(name) == safe, name


def test_generated_names_roundtrip():
    for name in GENERATED_NAMES:
        assert from_safe_name(to_safe_name(name)) == name, name


def test_generated_names_escape_once():
    for name in GENERATED_NAMES:
        safe = to_safe_name(name)
        assert to_safe_name(from_safe_name(safe)) == safe, name


def test_generated_legacy_names_convert():
    for name in GENERATED_NAMES:
        legacy = name.replace("/", "\\")
        assert to_safe_name(from_safe_name(legacy)) == to_safe_name(name), name


@pytest.mark.parametrize("character", ALL_CHARACTERS)
def test_every_character_roundtrip(character):
    name = f"a{character}b"
    safe = to_safe_name(name)
    assert re.fullmatch(r"[a-zA-Z0-9\-_~%]+", safe)
    assert from_safe_name(safe) == name


@pytest.mark.parametrize("character", ALL_CHARACTERS)
def test_validation_accepts_only_valid_characters(character):
    name = f"a{character}b"
    if character in VALID_CHARACTERS:
        assert validate_safe_name(name) == to_safe_name(name)
    else:
        with pytest.raises(ValueError):
            validate_safe_name(name)


@pytest.mark.parametrize(
    "length, valid", [(0, False), (1, True), (128, True), (129, False)]
)
def test_validation_length(length, valid):
    name = "a" * length
    if valid:
        assert validate_safe_name(name) == name
    else:
        with pytest.raises(ValueError):
            validate_safe_name(name)


def test_validation_rejects_trailing_newline():
    with pytest.raises(ValueError):
        validate_safe_name("valid_name\n")


def test_validation():
    assert validate_safe_name("valid_name") == "valid_name"
    assert validate_safe_name("with-dash") == "with-dash"
    assert validate_safe_name("with.dot") == "with%2Edot"
    assert validate_safe_name("path/to/item") == "path%2Fto%2Fitem"


def test_validation_errors():
    with pytest.raises(ValueError):
        validate_safe_name("has space")

    with pytest.raises(ValueError):
        validate_safe_name("has@symbol")
