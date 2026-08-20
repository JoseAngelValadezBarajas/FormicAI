from __future__ import annotations

import importlib
import platform
from typing import Any


def collect_environment_metadata() -> dict[str, Any]:
    torch_module = _import_optional_module("torch")
    cuda_available: bool | None = None
    torch_cuda_version: str | None = None
    if torch_module is not None:
        cuda = getattr(torch_module, "cuda", None)
        if cuda is not None and hasattr(cuda, "is_available"):
            try:
                cuda_available = bool(cuda.is_available())
            except Exception:
                cuda_available = None
        version = getattr(torch_module, "version", None)
        torch_cuda_version = str(getattr(version, "cuda", None)) if getattr(version, "cuda", None) else None

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": _module_version("numpy"),
        "opencv": _module_version("cv2"),
        "ultralytics": _module_version("ultralytics"),
        "torch": _module_version("torch"),
        "cudaAvailable": cuda_available,
        "torchCudaVersion": torch_cuda_version,
    }


def _module_version(module_name: str) -> str | None:
    module = _import_optional_module(module_name)
    if module is None:
        return None
    version = getattr(module, "__version__", None)
    return str(version) if version is not None else None


def _import_optional_module(module_name: str) -> Any | None:
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None
