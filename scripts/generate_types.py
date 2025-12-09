#!/usr/bin/env python3
"""Generate TypeScript types from Pydantic models."""

import subprocess
from pathlib import Path


def generate_typescript_types():
    """Generate TypeScript types from the Pydantic models."""
    # Use pydantic2ts to generate types
    output_dir = Path(__file__).parent.parent / "neuracore_types"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "neuracore_types.ts"

    # Generate using pydantic2ts
    cmd = [
        "pydantic2ts",
        "--module",
        "neuracore_types/__init__.py",
        "--output",
        str(output_file),
        "--json2ts-cmd",
        "npx json2ts --inferStringEnumKeysFromValues --enableConstEnums false",
    ]

    print(f"Generating TypeScript types to {output_file}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ TypeScript types generated successfully")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"✗ Error generating types: {e}")
        print(e.stderr)
        raise

    # Create index file that exports everything
    index_file = output_dir / "index.ts"
    index_file.write_text(
        """// Auto-generated index file
export * from './neuracore_types';
"""
    )
    print(f"✓ Created {index_file}")


if __name__ == "__main__":
    generate_typescript_types()
