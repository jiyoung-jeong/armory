"""Optional launch diagnostics kept separate from backend selection."""

from __future__ import annotations

import subprocess
from typing import Any


def get_gpu_info() -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        gpu_info = result.stdout.strip().split(", ")
        return {
            "gpu_available": True,
            "gpu_name": gpu_info[0],
            "driver_version": gpu_info[1],
            "memory_total": gpu_info[2],
        }
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return {"gpu_available": False}


_SOFTWARE_GL_RENDERERS = ("llvmpipe", "softpipe", "swr", "software rasterizer")


def assert_egl_rendering() -> None:
    """Assert that EGL bound to the NVIDIA GPU rather than software rendering."""
    from OpenGL import GL

    renderer = GL.glGetString(GL.GL_RENDERER)
    if renderer is None:
        raise AssertionError("Could not query GL_RENDERER; EGL context may not be initialized.")
    renderer_str = renderer.decode() if isinstance(renderer, bytes) else str(renderer)
    if any(sw in renderer_str.lower() for sw in _SOFTWARE_GL_RENDERERS):
        raise AssertionError(
            f"EGL is using a software renderer ({renderer_str!r}) instead of the NVIDIA "
            "GPU. Check NVIDIA_DRIVER_CAPABILITIES and /usr/share/glvnd/egl_vendor.d/."
        )
