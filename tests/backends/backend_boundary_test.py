"""Import, compatibility, and serialization checks for the backend boundary."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from typing import get_type_hints

from armory.backends.types import PolicyResult


def test_policy_result_contract_has_the_three_engine_required_keys() -> None:
    assert set(get_type_hints(PolicyResult)) == {
        "actions",
        "noise",
        "rtc_prev_actions",
    }
    assert PolicyResult.__required_keys__ == {
        "actions",
        "noise",
        "rtc_prev_actions",
    }


def test_compatibility_imports_and_pickles_are_canonical_and_lightweight() -> None:
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
        import serve_utils
        from scripts import serve_utils as package_serve_utils

        from armory.backends.mock import MockPolicyFactory
        from armory.backends.registry import Gr00tPolicyFactory, OpenPiPolicyFactory
        from armory.backends.types import EnvMode, ModelFamily
        from gr00t_adapter.serve_factory import EnvMode as Gr00tEnvMode
        from openpi_adapter.serve_factory import EnvMode as OpenPiEnvMode

        assert serve.EnvMode is EnvMode
        assert serve.ModelFamily is ModelFamily
        assert OpenPiEnvMode is EnvMode
        assert Gr00tEnvMode is EnvMode

        aliases = (serve_utils, package_serve_utils)
        for shim in aliases:
            assert shim._OpenPiFactory is OpenPiPolicyFactory
            assert shim._Gr00tFactory is Gr00tPolicyFactory
            assert shim._MockPolicyFactory is MockPolicyFactory

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

        # Protocol-0 GLOBAL payloads model old pickles that recorded either
        # historical module name. Direct aliases must keep both loadable.
        assert pickle.loads(b"cserve_utils\\n_OpenPiFactory\\n.") is OpenPiPolicyFactory
        assert (
            pickle.loads(b"cscripts.serve_utils\\n_OpenPiFactory\\n.")
            is OpenPiPolicyFactory
        )
        assert pickle.loads(b"copenpi_adapter.serve_factory\\nEnvMode\\n.") is EnvMode
        assert pickle.loads(b"cserve\\nModelFamily\\n.") is ModelFamily

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
