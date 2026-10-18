#!/usr/bin/env python3
"""Scan SakanaAI/AI-CUDA-Engineer-Archive with the frozen stale-weight detector.

Downloads the three parquet splits (column-projected), streams row batches,
writes raw JSON. Does not retune the detector. CPU only.

Usage:
  python3 scripts/scan_aicuda_archive.py --out results/stale_weight
"""
from __future__ import annotations

import argparse, hashlib, json, os, sys, time, urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from stale_weight_detector import detect_stale_weight_cache

HF = "https://huggingface.co/datasets/SakanaAI/AI-CUDA-Engineer-Archive/resolve/main"
SPLITS = ("level_1", "level_2", "level_3")
COLS = [
    "Op_Name",
    "Level_ID",
    "Task_ID",
    "Kernel_Name",
    "CUDA_Code",
    "PyTorch_Code_Module",
    "Correct",
]


def _download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return
    tmp = dest + ".part"
    print(f"downloading {url}", flush=True)
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)


def _git_head(repo: str) -> str | None:
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/stale_weight")
    ap.add_argument("--cache", default="data/aicuda_archive")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.cache, exist_ok=True)

    try:
        import pyarrow.parquet as pq
    except ImportError:
        os.system(f"{sys.executable} -m pip install -q pyarrow")
        import pyarrow.parquet as pq

    freeze = _git_head(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    t0 = time.time()

    n_scanned = 0
    n_correct = 0
    n_correct_flagged = 0
    n_stateful = 0
    n_functional = 0
    n_stateful_flagged = 0
    n_parse_error = 0
    per_level = {}
    flagged = []
    # Control: unique reference implementations (hash of source text).
    refs = {}  # hash -> {flagged, stateful, level, task, op}

    for split in SPLITS:
        path = os.path.join(args.cache, f"{split}.parquet")
        _download(f"{HF}/{split}.parquet", path)
        pf = pq.ParquetFile(path)
        level_id = int(split.split("_")[1])
        per_level.setdefault(
            level_id,
            {
                "scanned": 0,
                "correct": 0,
                "correct_flagged": 0,
                "stateful": 0,
                "functional": 0,
                "stateful_flagged": 0,
            },
        )
        for batch in pf.iter_batches(batch_size=args.batch, columns=COLS):
            cols = {name: batch.column(name).to_pylist() for name in batch.schema.names}
            n = len(cols["CUDA_Code"])
            for i in range(n):
                cuda = cols["CUDA_Code"][i] or ""
                ref = cols["PyTorch_Code_Module"][i] or ""
                correct = bool(cols["Correct"][i])
                task = int(cols["Task_ID"][i]) if cols["Task_ID"][i] is not None else -1
                op = cols["Op_Name"][i]
                kname = cols["Kernel_Name"][i]
                lid = int(cols["Level_ID"][i]) if cols["Level_ID"][i] is not None else level_id
                kid = f"L{lid}_T{task}_{kname}_{n_scanned}"
                n_scanned += 1
                pl = per_level[lid]

                d = detect_stale_weight_cache(cuda)
                pl["scanned"] += 1
                if d.parse_kind == "syntax_error" and not d.findings:
                    n_parse_error += 1
                if correct:
                    n_correct += 1
                    pl["correct"] += 1
                    if d.stateful:
                        n_stateful += 1
                        pl["stateful"] += 1
                    else:
                        n_functional += 1
                        pl["functional"] += 1
                    if d.flagged:
                        n_correct_flagged += 1
                        pl["correct_flagged"] += 1
                        if d.stateful:
                            n_stateful_flagged += 1
                            pl["stateful_flagged"] += 1
                        flagged.append(
                            {
                                "id": kid,
                                "level": lid,
                                "task_id": task,
                                "op_name": op,
                                "kernel_name": kname,
                                "stateful": d.stateful,
                                "parse_kind": d.parse_kind,
                                "findings": d.as_dict()["findings"],
                            }
                        )

                rh = hashlib.sha256(ref.encode("utf-8", "replace")).hexdigest()
                if rh not in refs:
                    rd = detect_stale_weight_cache(ref)
                    refs[rh] = {
                        "hash": rh,
                        "level": lid,
                        "task_id": task,
                        "op_name": op,
                        "flagged": rd.flagged,
                        "stateful": rd.stateful,
                        "findings": rd.as_dict()["findings"],
                    }
            if n_scanned % 2000 == 0:
                print(f"scanned {n_scanned} ... flagged_correct={n_correct_flagged}", flush=True)

    n_refs = len(refs)
    n_refs_flagged = sum(1 for r in refs.values() if r["flagged"])
    prev = (n_correct_flagged / n_correct) if n_correct else None
    prev_stateful = (n_stateful_flagged / n_stateful) if n_stateful else None
    ctrl = (n_refs_flagged / n_refs) if n_refs else None

    summary = {
        "dataset": "SakanaAI/AI-CUDA-Engineer-Archive",
        "schema_note": (
            "Real schema uses Correct (bool), CUDA_Code, PyTorch_Code_Module; "
            "three splits level_1/2/3. No 'correct==True' lowercase field."
        ),
        "detector_freeze_commit": freeze,
        "seconds": time.time() - t0,
        "n_scanned": n_scanned,
        "n_correct": n_correct,
        "n_correct_flagged": n_correct_flagged,
        "prevalence_correct": prev,
        "n_stateful_correct": n_stateful,
        "n_functional_correct": n_functional,
        "n_stateful_flagged": n_stateful_flagged,
        "prevalence_stateful": prev_stateful,
        "n_cuda_syntax_error_unflagged": n_parse_error,
        "per_level": per_level,
        "control": {
            "n_unique_pytorch_module_refs": n_refs,
            "n_flagged": n_refs_flagged,
            "rate": ctrl,
            "flagged_refs": [r for r in refs.values() if r["flagged"]],
        },
    }
    json.dump(summary, open(os.path.join(args.out, "scan_summary.json"), "w"), indent=1)
    json.dump(flagged, open(os.path.join(args.out, "flagged_kernels.json"), "w"), indent=1)
    json.dump(
        {"n": n_refs, "flagged": n_refs_flagged, "refs": list(refs.values())},
        open(os.path.join(args.out, "control_refs.json"), "w"),
        indent=1,
    )
    print(json.dumps({k: summary[k] for k in summary if k != "per_level"}, indent=1))
    print("per_level", json.dumps(per_level, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
