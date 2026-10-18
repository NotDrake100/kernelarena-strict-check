#!/usr/bin/env python3
"""Run the frozen detector on the 16 labelled KernelArena entries.

Prints and writes results/cpu/kernelarena_detector_gate.json.
Fails the process if the validation gate is not met.
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from stale_weight_detector import detect_stale_weight_cache
from test_stale_weight_detector import STALE, CLEAN, SOL

def main():
    missing = [f"{t}/{m}" for t, m in STALE + CLEAN if not os.path.exists(f"{SOL}/{t}/{m}/kernel.py")]
    if missing:
        print("STOP: labelled source not recoverable:", missing)
        print("Run: bash scripts/fetch_solutions.sh kb")
        return 2
    rows = []
    stale_hits, clean_hits = [], []
    for label, pairs in (("stale", STALE), ("clean", CLEAN)):
        for t, m in pairs:
            src = open(f"{SOL}/{t}/{m}/kernel.py").read()
            d = detect_stale_weight_cache(src)
            rec = {"id": f"{t}/{m}", "label": label, **d.as_dict()}
            rows.append(rec)
            if d.flagged and label == "stale":
                stale_hits.append(rec["id"])
            if d.flagged and label == "clean":
                clean_hits.append(rec["id"])
    gate = {
        "stale_flagged": len(stale_hits),
        "stale_total": len(STALE),
        "clean_flagged": len(clean_hits),
        "clean_total": len(CLEAN),
        "stale_hits": stale_hits,
        "clean_hits": clean_hits,
        "passed": len(stale_hits) == len(STALE) and len(clean_hits) <= 1,
    }
    out = {"gate": gate, "rows": rows}
    os.makedirs("results/cpu", exist_ok=True)
    json.dump(out, open("results/cpu/kernelarena_detector_gate.json", "w"), indent=1)
    print(json.dumps(gate, indent=1))
    return 0 if gate["passed"] else 1

if __name__ == "__main__":
    sys.exit(main())
