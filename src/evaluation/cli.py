"""CLI argument plumbing shared by armory-evaluation entrypoints."""

from __future__ import annotations

import argparse
import pathlib
from typing import Self

import tyro

from evaluation.recording import JSONBaseModel


class JsonArgs(JSONBaseModel):
    """Pydantic args base that supports `--json-path` defaults overlaid by tyro CLI flags."""

    json_path: pathlib.Path | None = None

    @classmethod
    def from_cli(cls) -> Self:
        pre = argparse.ArgumentParser(add_help=False)
        pre.add_argument("--json-path", type=pathlib.Path, default=None)
        known, remaining = pre.parse_known_args()

        if known.json_path is not None:
            defaults = cls.from_json(known.json_path)
            return tyro.cli(cls, args=remaining, default=defaults)
        return tyro.cli(cls, args=remaining)
