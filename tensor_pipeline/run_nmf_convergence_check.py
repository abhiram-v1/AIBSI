"""Re-run rank-8 NMF with a larger iteration budget."""

from __future__ import annotations

from pathlib import Path

import run_decomposition_benchmark_cpu as cpu


original_nmf = cpu.benchmark.NMF


def converged_nmf(**kwargs):
    kwargs["max_iter"] = 5000
    kwargs["tol"] = 1e-5
    return original_nmf(**kwargs)


cpu.benchmark.NMF = converged_nmf
cpu.benchmark.METHODS = ("nmf",)
cpu.benchmark.OUTPUT = Path(
    r"C:\Projects\AIBSI\tensor_pipeline\outputs\nmf_rank8_convergence_check"
)


if __name__ == "__main__":
    cpu.benchmark.main()
