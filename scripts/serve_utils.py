"""Compatibility exports for the pre-package backend launch utilities.

New code should import from ``armory.backends``. Direct aliases are retained
here so both legacy pickle module names (``serve_utils`` and
``scripts.serve_utils``) continue resolving existing factory objects.
"""

from armory.backends import diagnostics as _diagnostics
from armory.backends import mock as _mock
from armory.backends import registry as _registry
from armory.backends.types import EnvMode
from evaluation.cli import JsonArgs
from gr00t_adapter.serve_factory import (
    create_gr00t_policy,
    get_gr00t_checkpoint_label,
    get_gr00t_model_dims,
    is_groot_model,
)
from openpi_adapter.serve_factory import create_policy, get_model_dims

INFERENCE_PROFILES = _mock.INFERENCE_PROFILES
ResolvedPolicy = _registry.ResolvedPolicy
_Gr00tFactory = _registry.Gr00tPolicyFactory
_MockPolicy = _mock.MockPolicy
_MockPolicyFactory = _mock.MockPolicyFactory
_OpenPiFactory = _registry.OpenPiPolicyFactory
_SOFTWARE_GL_RENDERERS = _diagnostics._SOFTWARE_GL_RENDERERS
assert_egl_rendering = _diagnostics.assert_egl_rendering
create_default_policy = _registry.create_default_policy
get_gpu_info = _diagnostics.get_gpu_info
resolve_policy = _registry.resolve_policy

__all__ = [
    "EnvMode",
    "INFERENCE_PROFILES",
    "JsonArgs",
    "ResolvedPolicy",
    "_Gr00tFactory",
    "_MockPolicy",
    "_MockPolicyFactory",
    "_OpenPiFactory",
    "_SOFTWARE_GL_RENDERERS",
    "assert_egl_rendering",
    "create_default_policy",
    "create_gr00t_policy",
    "create_policy",
    "get_gpu_info",
    "get_gr00t_checkpoint_label",
    "get_gr00t_model_dims",
    "get_model_dims",
    "is_groot_model",
    "resolve_policy",
]
