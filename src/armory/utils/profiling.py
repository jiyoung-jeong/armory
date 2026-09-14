"""Optional NVTX annotations; enable with ARMORY_NVTX=1 before server startup.

Uses an existing NVIDIA NVTX runtime without importing a model backend or
initializing CUDA. Normal runs neither load NVTX nor emit annotations.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import pathlib
import sys
from contextlib import contextmanager
from functools import lru_cache

_ENABLED = os.environ.get("ARMORY_NVTX") == "1"


@lru_cache(maxsize=1)
def _library():
    candidates = [
        str(pathlib.Path(root) / "nvidia/nvtx/lib/libnvToolsExt.so.1") for root in sys.path if root
    ]
    candidates.append(ctypes.util.find_library("nvToolsExt") or "libnvToolsExt.so.1")
    for path in candidates:
        try:
            lib = ctypes.CDLL(path)
        except OSError:
            continue
        lib.nvtxRangeStartA.argtypes = [ctypes.c_char_p]
        lib.nvtxRangeStartA.restype = ctypes.c_uint64
        lib.nvtxRangeEnd.argtypes = [ctypes.c_uint64]
        lib.nvtxRangeEnd.restype = None
        lib.nvtxMarkA.argtypes = [ctypes.c_char_p]
        lib.nvtxMarkA.restype = None
        return lib
    raise RuntimeError("ARMORY_NVTX=1 requires an installed NVIDIA NVTX runtime")


@contextmanager
def nvtx_range(name: str):
    lib = _library() if _ENABLED else None
    if lib is not None:
        # Explicit IDs also keep overlapping asyncio tasks independent.
        range_id = lib.nvtxRangeStartA(name.encode())
    try:
        yield
    finally:
        if lib is not None:
            lib.nvtxRangeEnd(range_id)


def nvtx_mark(name: str) -> None:
    if _ENABLED:
        _library().nvtxMarkA(name.encode())
