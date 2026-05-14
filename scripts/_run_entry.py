"""Generic subprocess entrypoint: unpickle an Args dataclass and call <module>.main(args).

The Modal sweep containers pickle a ``serve.Args`` / ``run_libero.Args`` into the
run dir and exec this wrapper, rather than reconstructing a CLI argv. This is also
where the per-script multiprocessing start method is set: ``serve.py`` needs
``fork`` and ``run_libero.py`` needs ``spawn``, and the two conflict — so each has
to run in its own process anyway.

    python scripts/_run_entry.py <serve|run_libero> <args.pkl>
"""

from __future__ import annotations

import importlib
import multiprocessing as mp
import pathlib
import pickle
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

# entry name -> (module to import, multiprocessing start method)
ENTRYPOINTS = {
    "serve": ("serve", "fork"),
    "run_libero": ("run_libero", "spawn"),
}


def main() -> None:
    name, args_path = sys.argv[1], sys.argv[2]
    module_name, start_method = ENTRYPOINTS[name]
    mp.set_start_method(start_method, force=True)
    args = pickle.loads(pathlib.Path(args_path).read_bytes())
    importlib.import_module(module_name).main(args)


if __name__ == "__main__":
    main()
