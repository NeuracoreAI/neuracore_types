"""Tests for the TypeScript generated from the Pydantic models."""

import re
import runpy
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).parents[2]
GENERATE_TYPES_SCRIPT = REPOSITORY_ROOT / "scripts" / "generate_types.py"


def _interface_body(typescript: str, name: str) -> str:
    match = re.search(rf"export interface {name} \{{\n(.*?)\n\}}", typescript, re.S)
    assert match is not None, f"interface {name} is not generated"
    return match.group(1)


@pytest.fixture(scope="module")
def generated_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    generate_typescript_types = runpy.run_path(str(GENERATE_TYPES_SCRIPT))[
        "generate_typescript_types"
    ]
    output_dir = tmp_path_factory.mktemp("typescript")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.chdir(REPOSITORY_ROOT)
        generate_typescript_types(output_dir)
    return output_dir


@pytest.mark.parametrize(
    ("interface", "declaration"),
    [
        ("NCData", "timestamp_us?: number;"),
        ("JointData", "timestamp_us?: number;"),
        ("SynchronizedPoint", "timestamp_us?: number;"),
        ("SynchronizedEpisode", "start_timestamp_us?: number;"),
        ("SynchronizedEpisode", "end_timestamp_us?: number;"),
        ("Recording", "start_timestamp_us?: number | null;"),
        ("Recording", "end_timestamp_us?: number | null;"),
        ("RecordingStartRequest", "start_timestamp_us?: number | null;"),
        ("RecordingStopRequest", "end_timestamp_us?: number | null;"),
    ],
)
def test_microsecond_fields_are_optional(generated_dir, interface, declaration):
    typescript = (generated_dir / "neuracore_types.ts").read_text()
    assert f"  {declaration}" in _interface_body(typescript, interface).splitlines()


def test_index_re_exports_constants(generated_dir):
    index = (generated_dir / "index.ts").read_text()
    assert "export * from './constants';" in index.splitlines()
