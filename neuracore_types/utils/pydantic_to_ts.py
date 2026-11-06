"""Utility for fixing required fields with defaults."""

from pydantic.config import JsonDict

REQUIRED_WITH_DEFAULT_FLAG: JsonDict = {"REQUIRED_WITH_DEFAULT_FLAG": True}


def _pop_flagged_properties(properties: dict, flag: JsonDict) -> set[str]:
    """Remove a flag from each property schema and return the flagged names."""
    flag_key = next(iter(flag.keys()))
    return {
        name
        for name, config in properties.items()
        if isinstance(config, dict) and config.pop(flag_key, None)
    }


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

    flagged_properties = _pop_flagged_properties(properties, REQUIRED_WITH_DEFAULT_FLAG)

    exiting_required = json_schema.get("required", [])
    assert isinstance(exiting_required, list)
    json_schema["required"] = [*exiting_required, *flagged_properties]


OPTIONAL_COMPUTED_FIELD_FLAG: JsonDict = {"OPTIONAL_COMPUTED_FIELD_FLAG": True}


def fix_optional_computed_fields(json_schema: JsonDict) -> None:
    """Make flagged computed fields optional in the JSON schema.

    The serialization schema lists every computed field as required. Use this
    for a computed field that is new to a model, so that clients which build
    the model by hand do not have to set it.

    Usage
    @computed_field(json_schema_extra=OPTIONAL_COMPUTED_FIELD_FLAG)

    Args:
        json_schema: The JSON schema to modify.
    """
    properties = json_schema.get("properties", None)

    if not isinstance(properties, dict):
        return

    optional_properties = _pop_flagged_properties(
        properties, OPTIONAL_COMPUTED_FIELD_FLAG
    )

    required = json_schema.get("required", [])
    assert isinstance(required, list)
    json_schema["required"] = [
        name for name in required if name not in optional_properties
    ]
