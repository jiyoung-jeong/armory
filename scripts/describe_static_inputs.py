"""Record raw and model-boundary input specs by replaying preprocessing on CPU.

Does not load weights or run inference. The resulting JSON describes a post-hoc
reconstruction, not a tensor capture from the original timed GPU execution.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import urlparse


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(root, cache):
    import jax
    import numpy as np
    from openpi import transforms
    from openpi.training import config
    from scripts.benchmark_static_inference import load_inputs

    from openpi_adapter.policy_adapter import OpenPiPolicyAdapter, _rename_keys

    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Describe inputs only after the benchmark completes")
    records, requests = load_inputs(Path(manifest["inputs"]))
    cfg = config.get_config(manifest["config"])
    checkpoint = Path(manifest["checkpoint"])
    assert all(device.platform == "cpu" for device in jax.devices())
    resolved = {}

    def local_asset(url, **_kwargs):
        parsed = urlparse(str(url))
        path = cache / parsed.netloc / parsed.path.lstrip("/") if parsed.scheme else Path(url)
        if not path.exists():
            raise FileNotFoundError(f"Required local asset is absent (no download): {path}")
        resolved[str(url)] = path.resolve()
        return path

    # Use the same factories, checkpoint normalization, and transform order as
    # create_trained_policy, while making remote fetches impossible.
    with patch("openpi.shared.download.maybe_download", local_asset):
        factory = dataclasses.replace(
            cfg.data,
            assets=dataclasses.replace(cfg.data.assets, assets_dir=str(checkpoint / "assets")),
        )
        data = factory.create(cfg.assets_dirs, cfg.model)
        assert data.norm_stats is not None
        transform = transforms.compose(
            [
                transforms.InjectDefaultPrompt(None),
                *data.data_transforms.inputs,
                transforms.Normalize(data.norm_stats, use_quantiles=data.use_quantile_norm),
                *data.model_transforms.inputs,
            ]
        )
    adapter = OpenPiPolicyAdapter(
        SimpleNamespace(_input_transform=transform, _is_pytorch_model=False),
        make_example_fn=lambda: {},
    )

    def spec(value):
        array = np.asarray(value)
        return dict(
            shape=list(array.shape), dtype=str(array.dtype), logical_nbytes=int(array.nbytes)
        )

    raw = []
    for record, request in zip(records, requests, strict=True):
        observation = request.observation
        arrays = {key: spec(value) for key, value in observation.items() if key != "prompt"}
        raw.append(
            dict(
                robot=record["robot"],
                task=record["task"],
                snapshot_sha256=record["sha256"],
                arrays=arrays,
                raw_array_bytes=sum(value["logical_nbytes"] for value in arrays.values()),
                prompt=observation["prompt"],
                prompt_utf8_bytes=len(observation["prompt"].encode("utf-8")),
            )
        )

    batches = []
    for size in manifest["batch_sizes"]:
        obs = adapter.create_batch_obs([_rename_keys(req.observation) for req in requests[:size]])
        tensors = {}
        for key, value in obs.images.items():
            tensors[f"images/{key}"] = spec(value)
        for key, value in obs.image_masks.items():
            tensors[f"image_masks/{key}"] = spec(value)
        for key in ["state", "tokenized_prompt", "tokenized_prompt_mask"]:
            tensors[key] = spec(getattr(obs, key))
        assert all(value["shape"][0] == size for value in tensors.values())
        valid = np.asarray(obs.tokenized_prompt_mask).sum(axis=1).tolist()
        masks = {key: np.asarray(value).tolist() for key, value in obs.image_masks.items()}
        batches.append(
            dict(
                batch_size=size,
                snapshot_indices=[r["robot"] for r in records[:size]],
                tensors=tensors,
                observation_logical_nbytes=sum(
                    value["logical_nbytes"] for value in tensors.values()
                ),
                valid_prompt_tokens_per_request=valid,
                image_masks=masks,
            )
        )
    first = batches[0]
    largest = max(batches, key=lambda batch: batch["batch_size"])
    for batch in batches:
        for key, value in batch["tensors"].items():
            assert value["shape"][1:] == first["tensors"][key]["shape"][1:]
            assert value["dtype"] == first["tensors"][key]["dtype"]
        assert (
            batch["valid_prompt_tokens_per_request"]
            == largest["valid_prompt_tokens_per_request"][: batch["batch_size"]]
        )
    assets = []
    for path in sorted(set(resolved.values())):
        if path.is_dir():
            path = path / "norm_stats.json"
        assets.append(dict(path=str(path), sha256=sha256(path)))
    result = dict(
        schema_version=1,
        method="Post-hoc CPU replay of saved observations through the original adapter preprocessing",
        captured_during_timed_gpu_run=False,
        weights_loaded=False,
        inference_executed=False,
        backend=str(jax.default_backend()),
        jax_enable_x64=bool(jax.config.jax_enable_x64),
        runtime_git_commit=manifest["git_commit"],
        submodules=manifest["submodules"],
        config=manifest["config"],
        checkpoint=manifest["checkpoint"],
        model_state_elements_by_dtype=manifest["model_state_elements_by_dtype"],
        max_token_len=cfg.model.max_token_len,
        model_action_dim=cfg.model.action_dim,
        action_horizon=cfg.model.action_horizon,
        denoising_steps=manifest["num_steps"],
        discrete_state_input=cfg.model.discrete_state_input,
        normalization="quantile" if data.use_quantile_norm else "mean_std",
        assets=assets,
        raw_requests=raw,
        batches=batches,
        notes=[
            "Only the leading batch dimension changes; each batch uses the same prefix of saved inputs.",
            "Image layout is NHWC; the third image slot is padding with a false image mask.",
            "Prompt lengths include special tokens; 200 is padded capacity, not actual token count.",
            "Logical array bytes exclude weights, activations, allocator overhead, RNG, and sampling noise.",
            "UTF-8 text bytes and array nbytes are not serialized request or network byte counts.",
            "CPU reconstruction uses JAX's default 32-bit mode; no GPU-boundary dump was saved in the timed run.",
        ],
    )
    destination = root / "input_spec.json"
    with destination.open("x") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    print(destination)
    for batch in batches:
        print(
            f"B={batch['batch_size']} observation_bytes={batch['observation_logical_nbytes']} "
            f"valid_tokens={batch['valid_prompt_tokens_per_request']}"
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Completed static benchmark output directory")
    parser.add_argument("--cache", type=Path, required=True, help="Existing OpenPI asset cache")
    args = parser.parse_args()
    # Set before importing JAX, Torch, or any model module. These are process-local.
    os.environ.update(
        CUDA_VISIBLE_DEVICES="",
        JAX_PLATFORMS="cpu",
        JAX_ENABLE_X64="0",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
    )
    describe(args.root, args.cache)


if __name__ == "__main__":
    main()
