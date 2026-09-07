"""Run one locally throttled rolling block at a time.

The command is intentionally stateful: it discovers completed audited blocks,
runs only the next block, audits it, and exits. Re-run the command to continue
without concurrent training or sustained high CPU load.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .audit import audit_run
from .evaluate import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--block", type=int, required=True, choices=range(7))
    args = parser.parse_args()
    start = 1113 + 22 * args.block
    end = min(start + 21, 1248)
    root = Path("empirical_runs/world_model_v1/outputs/dow30_serial") / f"seed_{args.seed}" / f"block_{args.block:02d}"
    if (root / "manifest.json").exists():
        audit_run(root)
        print(f"already audited: {root}")
        return
    config = yaml.safe_load(Path("empirical_runs/world_model_v1/config.yaml").read_text(encoding="utf-8"))
    config.update({"output_dir": str(root), "first_test": start, "last_test": end,
                   "block_size": 22, "seed": args.seed, "device": "cpu", "torch_threads": 2})
    path = root.parent / f"config_block_{args.block:02d}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    output = run(path)
    print(audit_run(output))


if __name__ == "__main__":
    main()
