#!/usr/bin/env python3
"""Scan ScalingIntelligence/kernelbench-samples with the frozen stale-weight detector.

Does not retune the detector. CPU-only static AST. Reads kernel source only after
the K2 prereg commit and after corpus selection from cards/schemas/row counts.

Usage:
  python3 scripts/scan_stale_weight_k2.py \\
      --root /tmp/k2/kb-samples-git \\
      --out results/stale_weight_k2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from k2_stateful import is_stateful_kernel
from stale_weight_detector import detect_stale_weight_cache

FREEZE = "f560e15571c1c49a47d177e4bd7a32c0540da28a"


def _git(repo: str, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", "replace")).hexdigest()


def _verify_freeze(repo: str) -> dict:
    det = os.path.join(repo, "stale_weight_detector.py")
    working = _sha256_file(det)
    frozen_blob = subprocess.check_output(
        ["git", "show", f"{FREEZE}:stale_weight_detector.py"], cwd=repo
    )
    frozen = hashlib.sha256(frozen_blob).hexdigest()
    diff = subprocess.check_output(
        ["git", "diff", FREEZE, "--", "stale_weight_detector.py"], cwd=repo, text=True
    )
    return {
        "freeze_commit": FREEZE,
        "working_sha256": working,
        "freeze_sha256": frozen,
        "diff_empty": diff == "",
        "identical": working == frozen and diff == "",
    }


def _iter_kernel_json(root: str):
    for dirpath, dirnames, files in os.walk(root):
        if ".git" in dirpath.split(os.sep):
            continue
        dirnames.sort()
        for fn in sorted(files):
            if fn == "kernel.json":
                yield os.path.join(dirpath, fn)


def _load_local_refs(repo: str) -> dict[str, tuple[str, str]]:
    refs_dir = os.path.join(repo, "refs")
    out = {}
    if not os.path.isdir(refs_dir):
        return out
    for fn in sorted(os.listdir(refs_dir)):
        if not fn.endswith(".py"):
            continue
        src = open(os.path.join(refs_dir, fn), encoding="utf-8", errors="replace").read()
        out[_sha256_text(src)] = (fn, src)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/tmp/k2/kb-samples-git")
    ap.add_argument("--out", default="results/stale_weight_k2")
    ap.add_argument("--repo", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    args = ap.parse_args()
    repo = os.path.abspath(args.repo)
    os.makedirs(args.out, exist_ok=True)

    freeze = _verify_freeze(repo)
    if not freeze["identical"]:
        print(json.dumps(freeze, indent=1))
        print("REFUSING: detector is not identical to freeze commit", FREEZE, file=sys.stderr)
        return 2

    t0 = time.time()
    n_scanned = 0
    n_correct = 0
    n_correct_stateful = 0
    n_correct_functional = 0
    n_flagged_correct_stateful = 0
    n_flagged_correct_any = 0
    n_detector_stateful_correct = 0
    n_empty_correct = 0
    n_parse_error_correct = 0
    per_split: dict = {}
    per_level: dict = {}
    per_model: dict = {}
    flagged = []

    for path in _iter_kernel_json(args.root):
        rel = os.path.relpath(path, args.root)
        try:
            obj = json.load(open(path))
        except Exception:
            continue
        src = obj.get("kernel") or ""
        correct = obj.get("correct") is True
        level = obj.get("level")
        model = obj.get("model_name") or ""
        problem = obj.get("problem_name") or ""
        problem_id = obj.get("problem_id")
        split = rel.split(os.sep)[0]
        n_scanned += 1

        d = detect_stale_weight_cache(src)
        stateful = is_stateful_kernel(src)

        def bump(bucket, key, **kw):
            bucket.setdefault(
                key,
                {
                    "scanned": 0,
                    "correct": 0,
                    "correct_stateful": 0,
                    "correct_functional": 0,
                    "correct_stateful_flagged": 0,
                },
            )
            for k, v in kw.items():
                bucket[key][k] += v

        bump(per_split, split, scanned=1)
        bump(per_level, str(level), scanned=1)
        bump(per_model, model or "unknown", scanned=1)

        if not correct:
            continue
        n_correct += 1
        bump(per_split, split, correct=1)
        bump(per_level, str(level), correct=1)
        bump(per_model, model or "unknown", correct=1)
        if not str(src).strip():
            n_empty_correct += 1
        if d.parse_kind == "syntax_error":
            n_parse_error_correct += 1
        if d.stateful:
            n_detector_stateful_correct += 1
        if stateful:
            n_correct_stateful += 1
            bump(per_split, split, correct_stateful=1)
            bump(per_level, str(level), correct_stateful=1)
            bump(per_model, model or "unknown", correct_stateful=1)
        else:
            n_correct_functional += 1
            bump(per_split, split, correct_functional=1)
            bump(per_level, str(level), correct_functional=1)
            bump(per_model, model or "unknown", correct_functional=1)
        if d.flagged:
            n_flagged_correct_any += 1
            row = {
                "id": rel.replace(os.sep, "/"),
                "split": split,
                "level": level,
                "model_name": model,
                "problem_id": problem_id,
                "problem_name": problem,
                "correct": True,
                "stateful": stateful,
                "detector_stateful": d.stateful,
                "parse_kind": d.parse_kind,
                "findings": d.as_dict()["findings"],
            }
            if stateful:
                n_flagged_correct_stateful += 1
                bump(per_split, split, correct_stateful_flagged=1)
                bump(per_level, str(level), correct_stateful_flagged=1)
                bump(per_model, model or "unknown", correct_stateful_flagged=1)
                flagged.append(row)
        if n_scanned % 5000 == 0:
            print(
                f"scanned {n_scanned} correct={n_correct} "
                f"stateful={n_correct_stateful} flagged={n_flagged_correct_stateful}",
                flush=True,
            )

    # Control: local refs/ plus unique K1 PyTorch_Code_Module texts (no corpus ref field).
    control_rows = []
    local = _load_local_refs(repo)
    seen_hashes = set()
    # Re-detect K1 refs by re-running detector on stored source if present; K1 dump
    # has no source, so we re-scan only local refs/ files plus any K1 hashes we can
    # match after hashing local files. For unmatched K1 hashes we keep the prior
    # measured flag from the K1 JSON only if we also re-read the K1 flagged list
    # (0 flagged). Record that K1 unique texts are reused as control texts by
    # re-detecting from this repo's refs/ and from ScalingIntelligence/KernelBench
    # only if present locally; otherwise use K1's already-measured 0/229 plus
    # local refs/.
    k1_path = os.path.join(repo, "results/stale_weight/control_refs.json")
    k1 = json.load(open(k1_path)) if os.path.exists(k1_path) else {"refs": []}
    for r in k1.get("refs", []):
        h = r.get("hash")
        if not h or h in seen_hashes:
            continue
        seen_hashes.add(h)
        # No source in K1 dump: reuse the already-measured frozen-detector result.
        control_rows.append(
            {
                "hash": h,
                "source": "k1_pytorch_code_module",
                "level": r.get("level"),
                "op_name": r.get("op_name"),
                "flagged": bool(r.get("flagged")),
                "stateful": bool(r.get("stateful")),
                "findings": r.get("findings") or [],
                "note": "reused K1 unique-ref measurement; source text not stored in K1 dump",
            }
        )
    for h, (fn, src) in local.items():
        d = detect_stale_weight_cache(src)
        rec = {
            "hash": h,
            "source": f"refs/{fn}",
            "flagged": d.flagged,
            "stateful": is_stateful_kernel(src),
            "findings": d.as_dict()["findings"],
        }
        if h in seen_hashes:
            rec["note"] = "also in K1 unique-ref set; local file re-detected"
            # replace the reused K1 row with the re-detected one
            control_rows = [c for c in control_rows if c.get("hash") != h]
        seen_hashes.add(h)
        control_rows.append(rec)

    n_ctrl = len(control_rows)
    n_ctrl_flagged = sum(1 for r in control_rows if r["flagged"])
    prev = (n_flagged_correct_stateful / n_correct_stateful) if n_correct_stateful else None
    ctrl = (n_ctrl_flagged / n_ctrl) if n_ctrl else None

    summary = {
        "dataset": "ScalingIntelligence/kernelbench-samples",
        "detector_freeze_commit": FREEZE,
        "freeze_check": freeze,
        "seconds": time.time() - t0,
        "n_scanned": n_scanned,
        "n_correct": n_correct,
        "n_correct_stateful": n_correct_stateful,
        "n_correct_functional": n_correct_functional,
        "n_detector_stateful_correct": n_detector_stateful_correct,
        "n_empty_correct": n_empty_correct,
        "n_parse_error_correct": n_parse_error_correct,
        "n_flagged_correct_any": n_flagged_correct_any,
        "n_flagged_correct_stateful": n_flagged_correct_stateful,
        "prevalence_correct_stateful": prev,
        "per_split": per_split,
        "per_level": per_level,
        "per_model": per_model,
        "control": {
            "n_unique_refs": n_ctrl,
            "n_flagged": n_ctrl_flagged,
            "rate": ctrl,
            "rule": (
                "corpus has no reference-model field; control = unique K1 "
                "PyTorch_Code_Module hashes plus this repo refs/ re-detected"
            ),
        },
    }
    json.dump(summary, open(os.path.join(args.out, "scan_summary.json"), "w"), indent=1)
    json.dump(flagged, open(os.path.join(args.out, "flagged_kernels.json"), "w"), indent=1)
    json.dump(
        {
            "n": n_ctrl,
            "flagged": n_ctrl_flagged,
            "rate": ctrl,
            "refs": control_rows,
        },
        open(os.path.join(args.out, "control_refs.json"), "w"),
        indent=1,
    )
    printable = {k: summary[k] for k in summary if k not in {"per_model"}}
    print(json.dumps(printable, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
