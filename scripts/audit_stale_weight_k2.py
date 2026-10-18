#!/usr/bin/env python3
"""Seeded (seed 0) precision-audit sample of K2 flagged correct-stateful kernels.

Freezes sample ids. Verdicts are filled by reading code (and optional CPU
dynamic confirmation). Does not retune the detector.
"""
from __future__ import annotations

import argparse
import json
import os
import random


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flagged", default="results/stale_weight_k2/flagged_kernels.json")
    ap.add_argument("--out", default="results/stale_weight_k2/audit_sample.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()
    flagged = json.load(open(args.flagged))
    rng = random.Random(args.seed)
    if len(flagged) <= args.n:
        sample = list(flagged)
        how = "all_flagged"
    else:
        sample = rng.sample(flagged, args.n)
        how = f"random.Random({args.seed}).sample"
    verdicts = []
    for row in sample:
        findings = row.get("findings") or [{}]
        verdicts.append(
            {
                "id": row["id"],
                "split": row.get("split"),
                "level": row.get("level"),
                "model_name": row.get("model_name"),
                "problem_name": row.get("problem_name"),
                "caching_line": findings[0].get("snippet"),
                "caching_line_no": findings[0].get("line"),
                "reason_from_detector": findings[0].get("reason"),
                "keys": findings[0].get("keys"),
                "would_be_wrong_after_inplace_weight_update": None,
                "audit_reason": None,
                "dynamic": "not_run",
            }
        )
    out = {
        "seed": args.seed,
        "n_flagged": len(flagged),
        "n_sampled": len(sample),
        "sample_rule": how,
        "n_yes": None,
        "precision": None,
        "verdicts": verdicts,
        "note": (
            "No flagged kernels to audit; precision is undefined (not a < 80% fail)."
            if not flagged
            else "Fill would_be_wrong_after_inplace_weight_update for each verdict."
        ),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps({k: out[k] for k in out if k != "verdicts"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
