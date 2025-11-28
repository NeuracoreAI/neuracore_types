#!/usr/bin/env python3
"""Generate compiled Python + TypeScript code from .proto files."""

import subprocess
from pathlib import Path


def compile_protos() -> None:
    root = Path(__file__).parent.parent  # repo root
    proto_file = root / "neuracore_types" / "neuracore_types.proto"

    py_output_dir = root / "neuracore_types"
    js_output_dir = root / "dist"
    ts_types_dir = root / "neuracore_types"

    # Ensure output dirs exist
    js_output_dir.mkdir(parents=True, exist_ok=True)
    ts_types_dir.mkdir(parents=True, exist_ok=True)

    # python/mypy generation
    python_cmd = [
        "python",
        "-m",
        "grpc_tools.protoc",
        "-I",
        str(proto_file.parent),
        f"--python_out={py_output_dir}",
        f"--mypy_out={py_output_dir}",
        str(proto_file),
    ]
    subprocess.run(python_cmd, check=True)

    # typeScript generation
    ts_cmd = [
        "npx",
        "protoc",
        "-I",
        str(proto_file.parent),
        f"--js_out=import_style=commonjs,binary:{js_output_dir}",
        f"--ts_out={ts_types_dir}",
        str(proto_file),
    ]
    subprocess.run(ts_cmd, check=True)


if __name__ == "__main__":
    compile_protos()
