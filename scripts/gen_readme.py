"""Generate README.md; every number is read from results/*.json."""
import json,glob,re
G=[json.load(open(f)) for f in sorted(glob.glob('results/gpu/level*.json'))]
ENV=json.load(open('results/gpu/env.json')); GPU=ENV['gpu']
REPRO=json.load(open('results/gpu/fp8_hip_input_path_repro.json'))['runs']
AUD=json.load(open('results/cpu/static_audit_summary.json'))
UT=open('results/cpu/test_strict_checker_output.txt').read().strip().splitlines()[-1]
lab=lambda r:f"`{r['task']}` / `{r['model']}`"
stale=[r for r in G if r['strict_fixed']['passed'] and not r['strict_inplace']['passed']]
bad_fixed=[r for r in G if not r['strict_fixed']['passed']]
ok=[r for r in G if r['strict_fixed']['passed'] and r['strict_inplace']['passed']]
n=len(G); nchk=G[0]['strict_fixed']['n_checks']
f3=lambda x:f'{x:.3g}'
rows='\n'.join(f"| {lab(r)} | {'pass' if r['claimed']['passed'] else 'fail'} | {'pass' if r['strict_fixed']['passed'] else 'FAIL'} ({r['strict_fixed']['n_failures']}/{r['strict_fixed']['n_checks']} failed) | {'pass' if r['strict_inplace']['passed'] else 'FAIL'} ({r['strict_inplace']['n_failures']}/{r['strict_inplace']['n_checks']} failed, max err {f3(r['strict_inplace']['max_abs_err_max'])}) | {r['port_note']} |" for r in G)
g98=[r for r in G if r['task'].startswith('level2_98')]
g98s=[r['timing_ms']['measured_speedup'] for r in g98]; g98c=[r['claimed']['speedup'] for r in g98]
mg=[r for r in G if r['task'].startswith('level3_44')][0]
sd=[r for r in G if r['task'].startswith('level3_43') and 'SDPA' in r['port_note']]
fp=[r for r in G if r['task'].startswith('level4_104') and r['model']=='opus_4.5'][0]
rmin=min(x['max_abs_err'] for x in REPRO); rmax=max(x['max_abs_err'] for x in REPRO); rel=max(x['rel_fro_err'] for x in REPRO)
md=f"""# kernelarena-strict-check

A stricter correctness check for [KernelArena](https://kernelarena.ai/eval?suite=kernelbench-hip) submissions
(the KernelBench HIP / MI300X suite published by Wafer in [wafer-ai/wafer-data](https://github.com/wafer-ai/wafer-data)),
and what it finds when you re-run the passing submissions on an NVIDIA GPU.

## The problem in one paragraph

The leaderboard checks each submission with one `torch.allclose` (rtol = atol = 1e-3) against the PyTorch reference.
Several "passing" submissions cache work derived from the weights (fused conv+BN weights, pooled Linear weights,
dequantized FP8 weights) and only rebuild the cache when the tensor's memory address, `id()`, device or dtype changes.
When the weights are updated in place (an optimizer step, `load_state_dict`, `copy_`), the address stays the same,
the cache goes stale, and the model silently returns wrong answers. A single-call check with fixed weights cannot see this.

## What I ran

On **{GPU}** (PyTorch {ENV['torch']}, CUDA {ENV['cuda']}, TF32 off), for {n} passing submissions that can run on NVIDIA,
at the full KernelBench shapes plus edge-case batch sizes, with 3 seeds and 3 calls per shape ({nchk} checks per mode):

1. **Strict, fixed weights:** fresh random inputs on every call; input-mutation and output-aliasing checks.
2. **Strict, in-place weight updates:** same, but the weights of both models get the same in-place update between calls.
3. **Timing:** median of 10 runs (after 3 warm-up runs) of the reference vs the submission.

HIP code cannot run on NVIDIA. Where a submission's custom HIP function was small and pure (GELU / scale / max),
it was replaced by a PyTorch stand-in with the same formula, copied from the HIP source; everything else in the file
(including the caching logic) ran unchanged. The note column says exactly what was ported for each row.

## Results ({GPU})

| Submission | Leaderboard | Strict, fixed weights | Strict, in-place updates | What ran |
|---|---|---|---|---|
{rows}

![pass/fail](charts/pass_fail.png)

**Stale weight caches: {len(stale)} of {n}** submissions pass every fixed-weight check and then fail after an in-place weight update:

""" + '\n'.join(f"- {lab(r)}: max error with fixed weights {f3(r['strict_fixed']['max_abs_err_max'])}, after in-place update {f3(r['strict_inplace']['max_abs_err_max'])} ({r['strict_inplace']['n_failures']}/{r['strict_inplace']['n_checks']} checks fail)" for r in stale) + f"""

![stale cache error](charts/stale_cache_error.png)

**Fails even with fixed weights: {len(bad_fixed)}** ({', '.join(lab(r) for r in bad_fixed)}; max error {', '.join(f3(r['strict_fixed']['max_abs_err_max']) for r in bad_fixed)}). I did not investigate the root cause.

**Clean: {len(ok)} of {n}** pass both strict modes.

### FP8 input path that never rounds to FP8 (reproduction)

In `level4_104_FP8_Matmul` / `opus_4.5`, the HIP function `fast_fp8_quant_dequant_hip` scales, clamps and unscales in fp16
but never casts to an FP8 type, so the input is returned unchanged. Reproducing that math in PyTorch at {REPRO[0]['shape']} on {GPU} and comparing to real FP8
quantize-dequantize gives max error {rmin:.3f}–{rmax:.3f} (relative error {rel*100:.2f}%) over {len(REPRO)} seeds; it fails allclose at both 1e-3 and 1e-2.
This is a reproduction of the HIP math, not the HIP kernel itself.

## Timing: NOT comparable to the leaderboard (different hardware)

**Do not compare these numbers with the leaderboard speedups.** The leaderboard was timed on AMD MI300X with the real HIP kernels.
These timings are from {GPU}, where HIP code could not run: some HIP functions were replaced by PyTorch stand-ins,
one submission ran its own PyTorch fallback, and `torch.compile` fell back to eager for FP8. They only describe how the non-HIP
parts of each file behave on this NVIDIA GPU, and are not a measure of the submissions' real speed. An earlier version of this
README plotted them next to the leaderboard numbers; that chart was removed because the comparison was misleading.

Median of 10 runs after 3 warm-ups, CUDA events, ms:

| Submission | Reference (ms) | Submission as run here (ms) | Ratio on {GPU} | What ran |
|---|---|---|---|---|
""" + '\n'.join(f"| {lab(r)} | {r['timing_ms']['reference']:.2f} | {r['timing_ms']['modelnew']:.2f} | {r['timing_ms']['measured_speedup']:.2f}x | {r['port_note']} |" for r in G) + f"""

## The fix

`strict_checker.py` (pure PyTorch, device-agnostic): fresh inputs per call, in-place weight updates between calls, several
seeds, edge-case shapes, input-mutation and output-aliasing checks, plus `static_audit()` that tells you whether
`ModelNew.forward` calls custom code at all. Unit tests on CPU: `{UT}` (stale weight cache, cached output,
shape-specialised output, input mutation, honest model, and three audit cases).

For submissions, caching weight-derived tensors is fine if the cache key is the parameter's in-place version counter
(`weight._version`) rather than its address (suggested fix; not benchmarked here).

## Static audit of all submissions (CPU)

Of {AUD['files']} submission files, {AUD['passed']} passed on the leaderboard; the audit finds {AUD['verdicts_passed']['no_custom_kernel']} passes with no custom kernel code at all
({AUD['verdicts_passed_speedup_gt1']['no_custom_kernel']} of them with speedup > 1). The "custom kernel unused in forward" count in
`results/cpu/static_audit_summary.json` is a heuristic and overcounts (it misses kernels called through helpers), so treat it as unreliable.

## Limits

- **Everything here ran on NVIDIA ({GPU}) or CPU, not on MI300X.** Speedups on another GPU are not the leaderboard's speedups;
  they only show how much speedup the non-HIP parts give on that GPU.
- HIP kernels were not compiled. Only small pure functions were replaced by PyTorch stand-ins (listed per row).
- Wafer's level-4 reference implementations are not public. For `level4_104_FP8_Matmul` the reference is an **assumed**
  FP8 quantize-dequantize of input and current weight. Other level-4 passes (101, 103, 105) were not checked for that reason.
- Level 1–3 references are the upstream [KernelBench](https://github.com/ScalingIntelligence/KernelBench) files in `refs/` (MIT license).
- In-place weight updates are a stricter condition than the leaderboard's own protocol. They matter in real use (fine-tuning, weight reloading, serving updates).

## Reproduce

```bash
pip install torch pytest matplotlib
python -m pytest -q test_strict_checker.py
bash scripts/fetch_solutions.sh kb             # the exact submissions, pinned wafer-data commit
python gpu_run.py --sol kb/solutions --out results/gpu
python scripts/make_charts.py results/gpu charts
python scripts/gen_readme.py
```
"""
open('README.md','w').write(md); print('README chars',len(md))
