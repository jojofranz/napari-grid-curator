"""Interactive batch annotation tool for microscopy datasets with tiled grid visualization."""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._reader import napari_get_reader
from ._widget import ExclusionWidget

__all__ = (
    "napari_get_reader",
    "ExclusionWidget",
    "__version__",
)
