"""Model architectures and helpers for loading versioned model modules."""

from importlib import import_module
from types import ModuleType


def model_module_name(module_name: str) -> str:
    """Return the package-qualified name for a model module.

    Older checkpoints store names such as ``model_V10`` and
    ``sdf_model_V07``. Keep accepting those names after the model files have
    moved into this package.
    """
    if not isinstance(module_name, str) or not module_name.strip():
        raise ValueError("Model module name must be a non-empty string.")

    module_name = module_name.strip()
    if module_name == __name__ or module_name.startswith(f"{__name__}."):
        return module_name
    if "." not in module_name:
        return f"{__name__}.{module_name}"
    return module_name


def import_model_module(module_name: str) -> ModuleType:
    """Import a model module by its current or legacy checkpoint name."""
    return import_module(model_module_name(module_name))
