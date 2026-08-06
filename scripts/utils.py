import argparse
import pathlib
from typing import Self

import tyro

from evaluation.types import JSONBaseModel


class JsonArgs(JSONBaseModel):
    json_path: pathlib.Path | None = None

    @classmethod
    def from_cli(cls) -> Self:
        pre = argparse.ArgumentParser(add_help=False)
        pre.add_argument("--json-path", type=pathlib.Path, default=None)
        known, remaining = pre.parse_known_args()

        if known.json_path is not None:
            return tyro.cli(cls, args=remaining, default=cls.from_json(known.json_path))
        return tyro.cli(cls, args=remaining)
