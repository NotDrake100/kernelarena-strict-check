# Results: prevalence of stale-weight caching in AI-CUDA-Engineer-Archive

**Verdict: FAIL (prevalence < 5%).**

The frozen detector flagged **0 / 16459** kernels marked `Correct` (prevalence **0.0%**).
The pre-registered claim required ≥ 5%. Other kill-rule conditions did not fail:

| Condition | Result | Threshold |
|---|---|---|
| Validation gate (KernelArena labels) | PASS: 6/6 stale flagged, 0/10 clean flagged | all 6 stale, ≤ 1/10 clean |
| Prevalence among `Correct==True` | **FAIL: 0/16459 = 0.0%** | ≥ 5% |
| Audit precision (seed 0) | N/A: 0 flagged, sample empty | ≥ 80% of sampled flags |
| Control on unique PyTorch refs | PASS: 0/229 = 0.0% | ≤ 1% |

No retuning after archive results. Every count below was measured.

---

## Frozen hashes

| Item | Git commit |
|---|---|
| Pre-registration (`PREREG_stale_weight_prevalence.md`) | `441fe98290d786c7350be152c4837e120334a68a` |
| Detector freeze (after KernelArena gate, before scan) | `f560e15571c1c49a47d177e4bd7a32c0540da28a` |

Scan metadata records the freeze hash in `results/stale_weight/scan_summary.json`.

---

## Schema actually used

The Hugging Face dataset `SakanaAI/AI-CUDA-Engineer-Archive` (splits `level_1`, `level_2`, `level_3`) does **not** have a field named `correct`. The boolean field is **`Correct`**. Generated kernel source is **`CUDA_Code`**. The original non-CUDA model is **`PyTorch_Code_Module`**.

`CUDA_Code` is almost always a C++/CUDA torch extension (`#include <torch/extension.h>`, `torch::Tensor forward(...)`), not a Python `nn.Module`. Among the 16459 correct kernels, **16459 failed `ast.parse`** (all of them). 16323 contain `__global__` or `torch::Tensor`.

That is why the KernelArena-validated Python detector, plus a conservative C++ static-cache check, flagged nothing in this archive: the archive kernels take weight tensors as **function arguments on every call**, which is not the Python instance-cache bug the labels came from.

---

## Archive counts (measured)

| Quantity | Count |
|---|---:|
| Total kernels scanned | 30615 |
| `Correct == True` | 16459 |
| Flagged among correct | 0 |
| Prevalence among correct | 0 / 16459 = **0.0%** |
| nn.Module-style (stateful) among correct | 0 |
| Purely functional among correct | 16459 |
| Flagged among stateful | 0 |
| Prevalence among stateful | undefined (0 stateful CUDA_Code rows) |

Stateful vs functional used the pre-specified rule: a Python `nn.Module` subclass that constructs parameters/layer submodules. Raw C++ `forward(x, weight, …)` is functional.

### Per-level breakdown

| Level | Scanned | Correct | Flagged | Stateful (correct) | Functional (correct) |
|---:|---:|---:|---:|---:|---:|
| 1 | 12157 | 7222 | 0 | 0 | 7222 |
| 2 | 12938 | 6988 | 0 | 0 | 6988 |
| 3 | 5520 | 2249 | 0 | 0 | 2249 |

---

## Control (PyTorch references)

Unique `PyTorch_Code_Module` texts (SHA-256), one per distinct KernelBench-style reference:

| Quantity | Count |
|---|---:|
| Unique references | 229 |
| Flagged | 0 |
| Rate | 0 / 229 = **0.0%** (≤ 1%, control passes) |

Raw dump: `results/stale_weight/control_refs.json`.

---

## Precision audit (seed 0)

`random.Random(0)` was prepared to sample 30 flagged ids, or all flagged if fewer than 30.

| Quantity | Value |
|---|---|
| Flagged pool | 0 |
| Sample size | 0 (all of them) |
| Yes (would be wrong after in-place weight update) | 0 |
| Precision | **undefined** (empty sample; not a < 80% fail) |

Raw dump: `results/stale_weight/audit_sample.json`.
Flagged ids: `results/stale_weight/flagged_kernels.json` (empty list).

### Three example flagged archive kernels

None exist. The frozen detector produced an empty flag list.

The six KernelArena labelled stale entries the detector *did* flag (validation set, not the archive) are recorded in `results/cpu/kernelarena_detector_gate.json`. Caching lines:

1. `level2_73_Conv2d_BatchNorm_Scaling/gemini_3_flash` — `self.fused_weight = w * fused_scale` (keyed on `id()` / is-None)
2. `level2_98_Matmul_AvgPool_GELU_Scale_Max/grok_4.1_min` — `self.register_buffer('Wgroup', self.linear.weight.view(...).mean(dim=1))` (built once in `__init__`)
3. `level4_104_FP8_Matmul/opus_4.5` — `self._cached_weight_dequant = w_fp8.to(dtype) / w_scale` (keyed on `weight.data_ptr()`)

---

## Validation gate (measured before the freeze)

Source recovered from `scripts/fetch_solutions.sh` (wafer-data `7be29c1ea2a6681009b6e62ea778cba1d97d4a82`). Labels from `results/gpu/*.json`.

- Stale flagged: **6 / 6**
- Clean flagged: **0 / 10**
- Unit tests: 12 passed (`python3 -m pytest -q test_stale_weight_detector.py`)

Baseline KernelArena rate remains 6/16 = 37.5%.

---

## What was inspected but not used to retune

After seeing 0 flags, the archive text was counted (not used to change the detector):

- 416 correct kernels contain a `__constant__` symbol whose name looks like a weight/kernel/filter. Spot checks show `cudaMemcpyToSymbol` inside `forward` every call (not a stale instance cache).
- 76 correct kernels contain `static at::Tensor` / `static float*`. Spot checks show C++ internal-linkage **functions** (`static at::Tensor conv_transpose_bn_act(...)`), not static local weight caches.
- 107 correct kernels mention `data_ptr` and the word `cached`. Spot checks are comments (`// Using __ldg for cached read`, `// cached scalar value`), not pointer-keyed caches.

These observations are why a looser C++ heuristic would have been tempting and why the freeze rule forbids retuning: they are not the labelled Python bug, and loosening after seeing 0% would invalidate the claim.

---

## Kill-rule application

```
FAIL because prevalence 0.0% < 5%.
```

The detector is not too loose on the control (0.0% ≤ 1%). It is too *narrow for this archive's C++ functional schema* to support the ≥ 5% claim. That is a failed claim, not a failed measurement.
