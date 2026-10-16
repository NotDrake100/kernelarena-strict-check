# kernelarena-strict-check

## The question

Can AI write GPU kernels that are both faster and still correct? [KernelArena](https://kernelarena.ai/eval?suite=kernelbench-hip) ranks kernels written by AI models. To count, each kernel has to give the same output as the original model it replaces.

## The problem

Some entries save a copy of the weights once and keep reusing it. That is fine until the weights are updated in place, for example during training or when new weights are loaded. The entry doesn't notice the change, keeps using its old copy, and gives wrong answers.

The normal check never sees this, because it runs with the same weights the whole time.

## How we found it

We wrote a stricter check that changes the weights between calls and compares the output again each time.

Of the 16 entries we tested, 6 passed every normal check and then failed once the weights changed, and some were off by as much as 59. For each of these 6, 24 of the 27 checks failed. One more entry, `level3_43` / `kimi_k2`, fails even when the weights stay fixed.

![Pass and fail for each entry](charts/pass_fail.png)

This chart shows how big the error gets after the weights change. With the weights left alone the error is tiny. After the change it jumps to between 5.8 and 59.

![Error after the weights change](charts/stale_cache_error.png)

## The fix

The fix is to refresh the saved weights whenever they change.

## What this shows

The normal check missed this flaw. This kind of bug, where a kernel keeps old state and only breaks when something changes between calls, can be found with stricter checks like this one, and the same approach can be taken further across the leaderboard.

## Timing: NOT comparable

The leaderboard's speed numbers come from an AMD MI300X. We ran on an NVIDIA A100, and the AMD-specific kernel code can't run there, so our timings can't be compared with the leaderboard's.

## How to run

```bash
pip install torch pytest matplotlib
python -m pytest -q test_strict_checker.py
bash scripts/fetch_solutions.sh kb             # the exact entries, pinned wafer-data commit
python gpu_run.py --sol kb/solutions --out results/gpu
python scripts/make_charts.py results/gpu charts
```

Results are in `results/gpu/` (A100 run) and `results/cpu/`.
