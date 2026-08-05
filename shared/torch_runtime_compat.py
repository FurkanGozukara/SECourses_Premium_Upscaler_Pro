"""Early compatibility fixes for the app's PyTorch/Diffusers runtimes."""

from __future__ import annotations

import enum
import logging
from typing import Any


_DIFFUSERS_TORCHAO_LOGGER = "diffusers.quantizers.torchao.torchao_quantizer"
_NATIVE_ENUM_COMPAT_MARKER = "_torchao_native_enum_pytree_compat"
_WARNING_FILTER_MARKER = "_torchao_checkpoint_warning_filtered"
_DIFFUSERS_TORCHAO_CHECKPOINT_WARNING = (
    "Unable to import `torchao` Tensor objects. This may affect loading "
    "checkpoints serialized with `torchao`"
)


def ensure_native_enum_pytree_compat() -> bool:
    """Prevent obsolete Enum PyTree registration on newer PyTorch versions.

    PyTorch 2.13 handles Enum subclasses natively as opaque values for
    ``torch.compile``. TorchAO 0.18 still calls ``register_constant`` for them,
    which is deprecated and will become an error. This wrapper skips only that
    obsolete call and delegates every non-Enum registration unchanged.
    """
    try:
        from torch._library.opaque_object import is_opaque_type
        from torch.utils import _pytree
    except (AttributeError, ImportError, ModuleNotFoundError):
        return False

    register_constant = _pytree.register_constant
    if getattr(register_constant, _NATIVE_ENUM_COMPAT_MARKER, False):
        return True

    class _EnumProbe(enum.Enum):
        VALUE = 1

    try:
        if not is_opaque_type(_EnumProbe):
            return False
    except (TypeError, RuntimeError):
        return False

    def register_constant_compat(cls: type[Any]) -> None:
        if isinstance(cls, type) and issubclass(cls, enum.Enum):
            try:
                if is_opaque_type(cls):
                    return None
            except (TypeError, RuntimeError):
                pass
        return register_constant(cls)

    setattr(register_constant_compat, _NATIVE_ENUM_COMPAT_MARKER, True)
    register_constant_compat._torchao_compat_original = register_constant
    _pytree.register_constant = register_constant_compat
    return True


class _IrrelevantTorchAoCheckpointWarningFilter(logging.Filter):
    """Hide a Diffusers warning for an unsupported checkpoint format."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage() != _DIFFUSERS_TORCHAO_CHECKPOINT_WARNING


def suppress_irrelevant_diffusers_torchao_warning() -> None:
    """Hide only Diffusers' import-time TorchAO-checkpoint warning.

    The app's loaders use regular SafeTensors, PyTorch state dictionaries, GGUF,
    or their own quantized caches. They never deserialize Diffusers checkpoints
    containing TorchAO tensor subclasses, so the warning is not actionable here.
    """
    logger = logging.getLogger(_DIFFUSERS_TORCHAO_LOGGER)
    if getattr(logger, _WARNING_FILTER_MARKER, False):
        return
    logger.addFilter(_IrrelevantTorchAoCheckpointWarningFilter())
    setattr(logger, _WARNING_FILTER_MARKER, True)


def configure_torch_runtime_compat() -> None:
    """Install compatibility behavior before Diffusers or TorchAO is imported."""
    ensure_native_enum_pytree_compat()
    suppress_irrelevant_diffusers_torchao_warning()
