#!/usr/bin/env python3
"""Generate compiled Python + TypeScript code from .proto files."""

import subprocess
from pathlib import Path


def compile_protos():
    root = Path(__file__).parent.parent
    proto_file = root / "neuracore_types" / "neuracore_types.proto"

    output_dir = root / "neuracore_types"

    # python/mypy generation
    python_cmd = [
        "python",
        "-m",
        "grpc_tools.protoc",
        "-I",
        str(proto_file.parent),
        f"--python_out={output_dir}",
        f"--mypy_out={output_dir}",
        str(proto_file),
    ]
    subprocess.run(python_cmd, check=True)

    # typeScript generation
    ts_cmd = [
        "npx",
        "protoc",
        "-I",
        str(proto_file.parent),
        f"--js_out=import_style=commonjs,binary:{output_dir}",
        f"--ts_out={output_dir}",
        str(proto_file),
    ]
    subprocess.run(ts_cmd, check=True)


if __name__ == "__main__":
    compile_protos()
