"""Launch upstream RIFE scripts with compatibility fixes for current dependencies."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import Any


def install_numpy_binary_fromstring_compat(numpy_module: Any = None) -> None:
    """Route removed binary ``fromstring`` calls to ``frombuffer`` on NumPy 2.x."""
    if numpy_module is None:
        import numpy as numpy_module

    if getattr(numpy_module, "_secourses_binary_fromstring_compat", False):
        return

    original = numpy_module.fromstring

    def fromstring_compat(value, dtype=float, count=-1, sep="", *, like=None):
        if not sep and not isinstance(value, str):
            try:
                return numpy_module.frombuffer(value, dtype=dtype, count=count)
            except (TypeError, ValueError):
                pass
        kwargs = {"dtype": dtype, "count": count, "sep": sep}
        if like is not None:
            kwargs["like"] = like
        return original(value, **kwargs)

    numpy_module.fromstring = fromstring_compat
    numpy_module._secourses_binary_fromstring_compat = True


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: rife_inference_wrapper.py <upstream-script> [arguments...]")
    target = Path(sys.argv[1]).resolve()
    if not target.is_file():
        raise FileNotFoundError(f"RIFE inference script not found: {target}")

    install_numpy_binary_fromstring_compat()
    target_parent = str(target.parent)
    if target_parent not in sys.path:
        sys.path.insert(0, target_parent)
    sys.argv = [str(target), *sys.argv[2:]]
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
