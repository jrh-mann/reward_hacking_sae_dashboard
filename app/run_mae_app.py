import argparse
import os
import pathlib
import sys

from typing import Dict


def _is_dir_empty(path: pathlib.Path) -> bool:
    if not path.exists():
        return True
    try:
        next(path.iterdir())
        return False
    except StopIteration:
        return True


def _bootstrap_cache(cache_dir: pathlib.Path) -> Dict[str, str]:
    """
    Ensure the cache directory exists and contains the MAE datasets.

    Returns a mapping of model keys to base directories inside the cache.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Lazy import to keep core runtime light if user already has data
    from huggingface_hub import snapshot_download

    model_map = {
        "gpt": {
            "repo_id": "andyrdt/maes-gpt-oss-20b",
            "hf_name": "openai/gpt-oss-20b",
            "dir": cache_dir / "maes-gpt-oss-20b",
        },
    }

    for key, spec in model_map.items():
        target_dir = spec["dir"]
        if _is_dir_empty(target_dir):
            target_dir.mkdir(parents=True, exist_ok=True)
            # Download the dataset snapshot into the target directory
            snapshot_download(
                repo_id=spec["repo_id"],
                repo_type="dataset",
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
            )

    return {k: str(v["dir"]) for k, v in model_map.items()}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the MAE Feature Explorer (open-source)."
    )
    parser.add_argument(
        "--max_activating_example_cache_dir",
        type=str,
        default=".mae_cache",
        help="Directory where MAE datasets are cached. Will be created and populated if empty.",
    )
    parser.add_argument("--port", type=int, default=7863, help="Server port")
    parser.add_argument(
        "--default_layer", type=int, default=11, help="Default layer to open"
    )
    parser.add_argument(
        "--default_trainer", type=int, default=0, help="Default trainer to open"
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cache_dir = pathlib.Path(args.max_activating_example_cache_dir).expanduser().resolve()

    model_dirs = _bootstrap_cache(cache_dir)

    # Build model specs for the server from the cache
    model_specs = {}
    if os.path.isdir(model_dirs.get("gpt", "")):
        model_specs["gpt"] = {
            "name": "openai/gpt-oss-20b",
            "base_dir": model_dirs["gpt"],
        }

    if not model_specs:
        raise SystemExit(
            "No valid model datasets found in cache. Expected maes-gpt-oss-20b/."
        )

    # Start the server
    from .server import run_server

    run_server(
        model_specs=model_specs,
        default_layer=args.default_layer,
        default_trainer=args.default_trainer,
        port=args.port,
    )


if __name__ == "__main__":
    main(sys.argv[1:])


