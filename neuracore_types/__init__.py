"""Neuracore Types - Shared type definitions for Neuracore."""

from neuracore_types.dataset import *  # noqa: F403
from neuracore_types.endpoints import *  # noqa: F403
from neuracore_types.episode import *  # noqa: F403
from neuracore_types.nc_data import *  # noqa: F403
from neuracore_types.synchronization import *  # noqa: F403
from neuracore_types.training import *  # noqa: F403
from neuracore_types.upload import *  # noqa: F403

TORCH_AVAILABLE = True
try:
    import torch  # noqa: F401

    del torch
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    from neuracore_types.batched_nc_data import *  # noqa: F403

__version__ = "5.0.0"
