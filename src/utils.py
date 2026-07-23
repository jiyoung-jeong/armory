import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed everything for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)


def assert_egl_rendering() -> None:
    """Assert that EGL bound to the NVIDIA GPU rather than falling back to software.

    MUJOCO_GL=egl silently falls back to Mesa's software `llvmpipe` renderer
    when libglvnd can't find the NVIDIA EGL ICD (see `_add_nvidia_egl_icd` in
    scripts/modal/images.py), which craters sim throughput with no error.
    """
    from OpenGL import GL

    SOFTWARE_GL_RENDERERS = ("llvmpipe", "softpipe", "swr", "software rasterizer")
    renderer = GL.glGetString(GL.GL_RENDERER)
    if renderer is None:
        raise AssertionError("Could not query GL_RENDERER; EGL context may not be initialized.")
    renderer_str = renderer.decode() if isinstance(renderer, bytes) else str(renderer)
    if any(sw in renderer_str.lower() for sw in SOFTWARE_GL_RENDERERS):
        raise AssertionError(
            f"EGL is using a software renderer ({renderer_str!r}) instead of the NVIDIA "
            "GPU. Check NVIDIA_DRIVER_CAPABILITIES and /usr/share/glvnd/egl_vendor.d/."
        )
