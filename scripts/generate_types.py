#!/usr/bin/env python3
"""Generate TypeScript types from Pydantic models and standalone enums."""

import inspect
import json
import os
import shutil
import typing
from enum import Enum
from pathlib import Path
from tempfile import mkdtemp
from typing import Annotated

from pydantic import BaseModel, TypeAdapter
from pydantic2ts.cli.script import (
    clean_output_file,
    clean_schema,
    extract_pydantic_models,
    generate_json_schema_v2,
    import_module,
    is_submodule,
)

MODULE_PATH = "neuracore_types/__init__.py"
JSON2TS_CMD = "npx json2ts --inferStringEnumKeysFromValues --enableConstEnums false"


def _is_standalone_enum(obj: object) -> bool:
    return inspect.isclass(obj) and issubclass(obj, Enum) and obj is not Enum


def extract_enums(module) -> list[type[Enum]]:
    """Recursively collect Enum classes reachable from `module`."""
    enums = []
    module_name = module.__name__

    for _, enum_cls in inspect.getmembers(module, _is_standalone_enum):
        enums.append(enum_cls)

    for _, submodule in inspect.getmembers(
        module, lambda obj: is_submodule(obj, module_name)
    ):
        enums.extend(extract_enums(submodule))

    return enums


def deduplicate_enums(enums: list[type[Enum]]) -> list[type[Enum]]:
    """Deduplicate by class identity (not `__name__`), in a deterministic order."""
    by_identity = {id(e): e for e in enums}
    return sorted(
        by_identity.values(), key=lambda c: f"{c.__module__}.{c.__qualname__}"
    )


def _find_enums_in_annotation(
    annotation: object, seen: set[object], found: set[type[Enum]]
) -> None:
    if annotation is None or annotation in seen:
        return

    if inspect.isclass(annotation):
        if issubclass(annotation, Enum) and annotation is not Enum:
            seen.add(annotation)
            found.add(annotation)
            return
        if issubclass(annotation, BaseModel):
            _find_enums_reachable_from_model(annotation, seen, found)
            return

    seen.add(annotation)

    if typing.get_origin(annotation) is Annotated:
        args = typing.get_args(annotation)
        if args:
            _find_enums_in_annotation(args[0], seen, found)
        return

    for arg in typing.get_args(annotation):
        _find_enums_in_annotation(arg, seen, found)


def _find_enums_reachable_from_model(
    model: type[BaseModel], seen: set[object], found: set[type[Enum]]
) -> None:
    if model in seen:
        return
    seen.add(model)
    for field in model.model_fields.values():
        _find_enums_in_annotation(field.annotation, seen, found)


def find_model_reachable_enums(models: list[type[BaseModel]]) -> set[type[Enum]]:
    """Collect the Enum classes actually reachable through `models`' fields."""
    seen: set[object] = set()
    found: set[type[Enum]] = set()
    for model in models:
        _find_enums_reachable_from_model(model, seen, found)
    return found


def add_standalone_enums(
    schema: dict, enums: list[type[Enum]], reachable_enums: set[type[Enum]]
) -> None:
    """Add enums with no BaseModel reference into the schema's `$defs`."""
    for enum_cls in enums:
        if enum_cls in reachable_enums:
            continue

        name = enum_cls.__name__
        suffix = 2
        while name in schema["$defs"]:
            name = f"{enum_cls.__name__}_{suffix}"
            suffix += 1

        enum_schema = TypeAdapter(enum_cls).json_schema(mode="serialization")
        clean_schema(enum_schema)
        schema["$defs"][name] = enum_schema

        prop_key = f"_standalone_enum_{name}"
        schema["properties"][prop_key] = {"$ref": f"#/$defs/{name}"}
        schema["required"].append(prop_key)


def generate_typescript_types():
    """Generate TypeScript types from the Pydantic models and standalone enums."""
    output_dir = Path(__file__).parent.parent / "neuracore_types"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "neuracore_types.ts"

    module = import_module(MODULE_PATH)
    models = extract_pydantic_models(module)
    enums = deduplicate_enums(extract_enums(module))
    reachable_enums = find_model_reachable_enums(models)

    schema = json.loads(generate_json_schema_v2(models))
    add_standalone_enums(schema, enums, reachable_enums)

    schema_dir = mkdtemp()
    schema_file_path = os.path.join(schema_dir, "schema.json")
    with open(schema_file_path, "w") as f:
        json.dump(schema, f, indent=2)

    print(f"Generating TypeScript types to {output_file}...")
    try:
        exit_code = os.system(
            f'{JSON2TS_CMD} -i {schema_file_path} -o {output_file} --bannerComment ""'
        )
    finally:
        shutil.rmtree(schema_dir)

    if exit_code != 0:
        raise RuntimeError(f'"{JSON2TS_CMD}" failed with exit code {exit_code}.')

    clean_output_file(str(output_file))
    print("✓ TypeScript types generated successfully")

    index_file = output_dir / "index.ts"
    index_file.write_text(
        """// Auto-generated index file
export * from './neuracore_types';
"""
    )
    print(f"✓ Created {index_file}")


if __name__ == "__main__":
    generate_typescript_types()
