"""Run one benchmark rank into a rank-specific output directory."""

from __future__ import annotations

import sys
from pathlib import Path

import run_decomposition_benchmark_cpu as cpu


def requested_rank(argv: list[str]) -> int:
    if "--rank" not in argv:
        return 8
    position = argv.index("--rank")
    return int(argv[position + 1])


rank = requested_rank(sys.argv)
REPO_ROOT = Path(__file__).resolve().parents[1]  # repository root (was a hard-coded C:\ path)
cpu.benchmark.OUTPUT = (
    REPO_ROOT / "tensor_pipeline" / "outputs" / "benchmark_rank_sweep"
    / f"rank_{rank}"
)


if __name__ == "__main__":
    cpu.benchmark.main()
