"""Import and serialization checks for the backend boundary."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path


def test_backend_imports_and_pickles_are_canonical_and_lightweight() -> None:
    repo_root = Path(__file__).parents[2]
    code = textwrap.dedent(
        """
        import importlib.abc
        import pathlib
        import pickle
        import sys
        import tempfile

        blocked = {"jax", "torch", "gr00t", "openpi", "libero", "robosuite"}

        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".", 1)[0] in blocked:
                    raise AssertionError(f"heavy dependency imported during resolution: {fullname}")
                return None

        sys.meta_path.insert(0, Blocker())
        sys.path.insert(0, "scripts")

        import serve
        from armory.backends.mock import MockPolicyFactory
        from armory.backends.registry import Gr00tPolicyFactory, OpenPiPolicyFactory
        from armory.backends.types import EnvMode, ModelFamily
        from gr00t_adapter.serve_factory import EnvMode as Gr00tEnvMode
        from openpi_adapter.serve_factory import EnvMode as OpenPiEnvMode

        assert serve.EnvMode is EnvMode
        assert serve.ModelFamily is ModelFamily
        assert OpenPiEnvMode is EnvMode
        assert Gr00tEnvMode is EnvMode

        factories = [
            OpenPiPolicyFactory("config", "/checkpoint", 10, EnvMode.LIBERO),
            Gr00tPolicyFactory("gr00t-n1.7", EnvMode.LIBERO, None),
            MockPolicyFactory(
                env="libero",
                action_horizon=20,
                action_dim=7,
                model="pi05",
                gpu="l40s",
            ),
        ]
        for factory in factories:
            restored = pickle.loads(pickle.dumps(factory))
            assert type(restored) is type(factory)
            assert vars(restored) == vars(factory)
            assert type(restored).__module__.startswith("armory.backends.")

        # Modal persists serve.Args between the launch and remote worker. Keep
        # the public enums and policy variants stable across that round trip.
        args = serve.Args(
            model=ModelFamily.GROOT_N17,
            env=EnvMode.LIBERO,
            policy=serve.Mock(
                action_horizon=20,
                action_dim=7,
                model="pi05",
                gpu="l40s",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "serve_args.json"
            args.to_json(path)
            restored_args = serve.Args.from_json(path)
        assert restored_args.model is ModelFamily.GROOT_N17
        assert restored_args.env is EnvMode.LIBERO
        assert isinstance(restored_args.policy, serve.Mock)

        assert not blocked.intersection(sys.modules)
        """
    )

    subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
