import importlib.util

_OPTIONAL_STACKS = {
    "jax": "backends/openpi_adapter_test.py",
    "fastapi": "serving/server_smoke_test.py",
}

collect_ignore = [
    path for module, path in _OPTIONAL_STACKS.items() if not importlib.util.find_spec(module)
]
