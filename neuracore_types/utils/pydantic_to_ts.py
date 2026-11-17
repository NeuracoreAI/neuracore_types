"""Utility for fixing required fields with defaults."""

from pydantic.config import JsonDict

REQUIRED_WITH_DEFAULT_FLAG: JsonDict = {"REQUIRED_WITH_DEFAULT_FLAG": True}


def fix_required_with_defaults(json_schema: JsonDict) -> None:
    """Helper function to fix required fields with defaults.

    Pydantic2ts generates optional fields for fields with default values.
    This function adds those fields to the required list in the JSON schema.


    Usage
    for every field with a default value,
    = Field(
        default_factory=list or default=, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )


    add this to models:
    model_config = ConfigDict(json_schema_extra=fix_required_with_defaults)


    Args:
        json_schema: The JSON schema to modify.
    """
    properties = json_schema.get("properties", None)

    if not isinstance(properties, dict):
        return

    flagged_properties = set()
    flag = next(iter(REQUIRED_WITH_DEFAULT_FLAG.keys()))
    for property, config in properties.items():
        if not isinstance(config, dict):
            continue
        if config.pop(flag, None):
            flagged_properties.add(property)

    exiting_required = json_schema.get("required", [])
    assert isinstance(exiting_required, list)
    json_schema["required"] = [*exiting_required, *flagged_properties]
