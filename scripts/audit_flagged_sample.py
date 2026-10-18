#!/usr/bin/env python3
"""Seeded (seed 0) precision-audit sample of flagged archive kernels.

Does not retune the detector. If fewer than 30 kernels were flagged, audit all.
Writes results/stale_weight/audit_sample.json.
"""
import argparse, json, os, random, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flagged", default="results/stale_weight/flagged_kernels.json")
    ap.add_argument("--out", default="results/stale_weight/audit_sample.json")
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
        # Verdicts are filled by a human/agent reading the code. This script
        # only freezes the sample ids. Empty flagged list => empty audit.
        verdicts.append(
            {
                "id": row["id"],
                "level": row.get("level"),
                "kernel_name": row.get("kernel_name"),
                "caching_line": (row.get("findings") or [{}])[0].get("snippet"),
                "would_be_wrong_after_inplace_weight_update": None,
                "reason": None,
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
            "No flagged kernels to audit; precision is undefined (not < 80%)."
            if not flagged
            else "Fill would_be_wrong_after_inplace_weight_update for each verdict."
        ),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps({k: out[k] for k in out if k != "verdicts"}, indent=1))
    return 0

if __name__ == "__main__":
    sys.exit(main())
