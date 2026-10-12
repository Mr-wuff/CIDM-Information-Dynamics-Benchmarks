"""Validate archived CIDM manuscript experiment tables.

Checks source-record counts and study identities, without performing training.
The tool does not claim independent numerical replication.
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STUDIES = {
    "controlled_comparison": {
        "file": "experiments/pde_benchmarks/controlled_comparison/results/seedwise_predictions.csv",
        "rows": 180,
        "tasks": 3,
        "models": 6,
        "seeds": 10,
        "identity": ("task", "model", "seed"),
    },
    "multiphysics_benchmark": {
        "file": "experiments/pde_benchmarks/multiphysics_benchmark/results/seedwise_predictions_publication.csv",
        "rows": 120,
        "tasks": 8,
        "models": 5,
        "seeds": 3,
        "identity": ("task", "model", "seed"),
    },
    "component_ablation": {
        "file": "experiments/pde_benchmarks/component_ablation/results/seedwise_ablation.csv",
        "rows": 110,
        "tasks": 5,
        "seeds": 3,
        "identity": ("task", "variant", "seed"),
    },
    "long_horizon_evaluation": {
        "file": "experiments/pde_benchmarks/long_horizon_evaluation/results/seedwise_training_and_evaluation.csv",
        "rows": 48,
        "tasks": 4,
        "models": 4,
        "seeds": 3,
        "identity": ("task", "model", "seed"),
    },
    "state_consistency": {
        "file": "experiments/pde_benchmarks/state_consistency/results/seedwise_training_and_evaluation.csv",
        "rows": 18,
        "tasks": 3,
        "models": 2,
        "seeds": 3,
        "identity": ("task", "model", "seed"),
    },
}


def validate_study(name: str, contract: dict) -> None:
    path = ROOT / contract["file"]
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == contract["rows"], (name, "row count", len(rows))
    assert len({r["task"] for r in rows}) == contract["tasks"], (name, "task count")
    assert len({r["seed"] for r in rows}) == contract["seeds"], (name, "seed count")
    if "models" in contract:
        assert len({r["model"] for r in rows}) == contract["models"], (name, "model count")

    identities = [tuple(r[k] for k in contract["identity"]) for r in rows]
    assert len(set(identities)) == len(rows), (name, "duplicate record identity")
    assert Counter(r["status"] for r in rows) == {"COMPLETED": len(rows)}, (name, "unfinished records")

    if name == "multiphysics_benchmark":
        corrected = [r for r in rows if r["task"] == "diff_sorp_1d"]
        assert len(corrected) == 15, "Expected 5 models × 3 corrected diffusion–sorption seeds"
        assert len(rows) - len(corrected) == 105

    print(f"PASS {name}: {len(rows)} completed independent experiment records")


if __name__ == "__main__":
    for name, contract in STUDIES.items():
        validate_study(name, contract)
    print("PASS all manuscript experimental record contracts (476 records total)")
