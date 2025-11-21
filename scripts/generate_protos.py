#!/usr/bin/env python3
"""Generate compiled Python + TypeScript code from .proto files."""

import subprocess
from pathlib import Path


def compile_protos():
    root = Path(__file__).parent.parent
    proto_file = root / "neuracore_types" / "neuracore_types.proto"

    python_out = root / "neuracore_types"
    ts_out = root / "typescript_protos"

    ts_out.mkdir(exist_ok=True)

    # python generation
    python_cmd = [
        "python",
        "-m",
        "grpc_tools.protoc",
        "-I",
        str(proto_file.parent),
        "--python_out",
        str(python_out),
        str(proto_file),
    ]
    subprocess.run(python_cmd, check=True)

    # typeScript generation
    ts_cmd = [
        "npx",
        "protoc",
        "-I",
        str(proto_file.parent),
        f"--js_out=import_style=commonjs,binary:{ts_out}",
        f"--ts_out={ts_out}",
        str(proto_file),
    ]
    subprocess.run(ts_cmd, check=True)


if __name__ == "__main__":
    compile_protos()
