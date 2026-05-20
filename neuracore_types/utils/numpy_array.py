"""Pydantic-compatible annotation for NumPy array fields.

Several models in this package store data as :class:`numpy.ndarray` and enable
``arbitrary_types_allowed`` so Pydantic can validate them. Pydantic cannot,
however, build a JSON schema for an arbitrary type, so any consumer that
generates a JSON schema (for example FastAPI's OpenAPI endpoint) fails with
``PydanticInvalidForJsonSchema`` for ``numpy.ndarray``.

Annotating array fields with :data:`NumpyArray` attaches an explicit JSON
schema, so schema generation succeeds. Validation and serialization behaviour
is unchanged: the existing ``field_validator``/``field_serializer`` hooks still
handle conversion between arrays and their JSON representation.
"""

from typing import Annotated

import numpy as np
from pydantic import WithJsonSchema

NumpyArray = Annotated[
    np.ndarray,
    WithJsonSchema({"type": "array", "items": {}}),
]
"""A ``numpy.ndarray`` field that exposes an ``array`` JSON schema."""
