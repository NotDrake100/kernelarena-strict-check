# Pre-registration: prevalence of stale-weight caching in AI-generated CUDA kernels

**Status:** pre-registered. This file is committed *before* the detector is frozen
and *before* any scan of `SakanaAI/AI-CUDA-Engineer-Archive`. Numbers below
are only the already-measured KernelArena baseline from this repository.
Archive counts will be written later to `RESULTS_stale_weight_prevalence.md`.

**Date:** 2026-10-18
**Repo:** kernelarena-strict-check (this repository)
**Archive:** Hugging Face dataset `SakanaAI/AI-CUDA-Engineer-Archive`
**Method:** CPU-only static analysis of source text / Python AST (plus reading
CUDA strings). No GPU rental. No CUDA execution of archive kernels.

---

## Claim (frozen wording)

A static detector for stale-weight caching (a module caches a derived copy of a
parameter/buffer, built once or keyed on `data_ptr()`/`id()`/shape/is-None, and
reuses it on later forward calls without checking the parameter's version or
values), frozen at a committed hash after validating on the labelled KernelArena
entries, flags >= 5% of the kernels marked correct in
`SakanaAI/AI-CUDA-Engineer-Archive` (Hugging Face dataset), with >= 80%
precision on a manual audit of a seeded random sample of 30 flagged kernels
(all of them if fewer than 30).

---

## What "stale-weight caching" means

A kernel (typically an `nn.Module` subclass) keeps a derived copy of a
parameter or buffer — for example a transpose, a fused conv+BN weight, a
packed / quantized / dequantized copy — and reuses that copy on later
`forward` calls. The rebuild trigger is one of:

- built once (`if self._cache is None`, or no check at all after `__init__`)
- keyed on `tensor.data_ptr()` / `id(tensor)` (in-place updates keep the same
  address and identity)
- keyed on shape / device / dtype / is-None (an in-place value change does
  not change those)

and **not** keyed on the parameter's in-place version (`tensor._version`) or
on the actual values.

Consequence: an in-place weight update (`add_`, `copy_`, optimizer step,
`load_state_dict` that writes into existing storage) leaves the cache stale,
so later outputs are wrong. A single-call correctness check with frozen
weights never sees this.

---

## Labels recovered from this repository (do not invent)

Source of labels: `results/gpu/level*.json` produced by `gpu_run.py` on the
16 KernelArena submissions listed in `gpu_run.py:CASES`. Kernel source is
**not** stored in this repo; it is recoverable from the pinned wafer-data
commit referenced by `scripts/fetch_solutions.sh`:

`wafer-ai/wafer-data` @ `7be29c1ea2a6681009b6e62ea778cba1d97d4a82`

**Stale** = `strict_fixed.passed == true` AND `strict_inplace.passed == false`
(passed every normal/fixed-weight check, then failed after in-place weight
updates). **Clean** for the validation gate = the other 10 of the 16 testable
entries (9 that passed both modes, plus `level3_43 / kimi_k2`, which fails
even with fixed weights and is therefore not a stale-weight miss of the
normal check).

### The 6 known stale entries

| id | task | model | fixed max abs err | inplace max abs err | inplace failures |
|---|---|---|---|---|---|
| S1 | `level2_73_Conv2d_BatchNorm_Scaling` | `gemini_3_flash` | 2.86e-6 | 26.5 | 24/27 |
| S2 | `level2_98_Matmul_AvgPool_GELU_Scale_Max` | `gemini_3_flash` | 1.10e-6 | 5.84 | 24/27 |
| S3 | `level2_98_Matmul_AvgPool_GELU_Scale_Max` | `gpt_5.2` | 1.07e-6 | 5.84 | 24/27 |
| S4 | `level2_98_Matmul_AvgPool_GELU_Scale_Max` | `gpt_5.2_min` | 1.11e-5 | 5.84 | 24/27 |
| S5 | `level2_98_Matmul_AvgPool_GELU_Scale_Max` | `grok_4.1_min` | 1.07e-6 | 5.84 | 24/27 |
| S6 | `level4_104_FP8_Matmul` | `opus_4.5` | 0.0 | 59.4 | 24/27 |

These match the README claim: 6 of 16, 24 of 27 checks failed, errors up to 59.

### The 10 known clean (non-stale) entries

| id | task | model | notes |
|---|---|---|---|
| C1 | `level1_9_Tall_skinny_matrix_multiplication_` | `gpt_5.2_min` | both strict modes pass |
| C2 | `level2_40_Matmul_Scaling_ResidualAdd` | `grok_4.1` | both strict modes pass |
| C3 | `level2_40_Matmul_Scaling_ResidualAdd` | `opus_4.5_min` | both strict modes pass |
| C4 | `level2_55_Matmul_MaxPool_Sum_Scale` | `grok_4.1_min` | both strict modes pass |
| C5 | `level3_43_MinGPTCausalAttention` | `grok_4.1_min` | both strict modes pass |
| C6 | `level3_43_MinGPTCausalAttention` | `opus_4.5` | both strict modes pass |
| C7 | `level3_43_MinGPTCausalAttention` | `opus_4.5_min` | both strict modes pass |
| C8 | `level3_44_MiniGPTBlock` | `gpt_5.2` | both strict modes pass |
| C9 | `level4_104_FP8_Matmul` | `gemini_3_flash` | both strict modes pass |
| C10 | `level3_43_MinGPTCausalAttention` | `kimi_k2` | fails even with fixed weights (max abs err 0.074); not a stale-weight miss of the normal check |

If the labelled `kernel.py` files cannot be recovered from the pinned
wafer-data commit (or another referenced source in this repo), **stop**. Do
not invent labels. Synthetic unit tests may be added later but do not replace
this gate.

---

## Baseline

KernelArena rate among the 16 testable entries: **6/16 = 37.5%**.

---

## Validation gate (must pass before the archive is scanned)

The detector is developed only against:

1. the 16 labelled KernelArena entries above (source fetched from the pinned
   commit), and
2. optional, clearly separate synthetic unit tests.

**Pass the gate iff:**

- all 6 known stale entries are flagged, AND
- at most 1 of the 10 known clean entries is flagged.

Then commit the detector. The commit hash of that freeze is the detector
version used on the archive. **No retuning after archive results are seen.**

---

## Archive protocol (executed only after the freeze commit)

1. Stream `SakanaAI/AI-CUDA-Engineer-Archive` (do not download more than
   needed). Adapt to the real schema if it differs; record the schema.
2. Restrict the prevalence denominator to rows marked correct
   (`correct == True`, or the dataset's equivalent field). Record the field
   name actually used.
3. Run the frozen detector on each such kernel's source (Python + any
   embedded CUDA string).
4. Classify each kernel as **nn.Module-style (stateful)** vs **purely
   functional** using a pre-specified static rule (presence of an
   `nn.Module` / `Module` subclass that holds parameters/buffers or
   instance caches, vs a lone function). Report prevalence on all
   correct kernels and on the stateful subset.
5. If the archive has levels (KernelBench-style level 1–4 or equivalent),
   report a per-level breakdown.
6. **Control:** run the same frozen detector on the archive's PyTorch
   reference implementations (the original non-CUDA model code). It must
   flag <= 1% of those references. If it flags more than 1%, the detector
   is too loose and the claim **FAILS**.
7. **Precision audit:** take a seeded random sample of 30 flagged
   *correct* kernels (`random.Random(0).sample`, seed 0; all flagged
   kernels if fewer than 30). For each, read the code and decide whether a
   real in-place weight update between calls would produce a wrong output
   (yes/no) with a one-line reason that quotes the caching line. Record
   the sample ids and verdicts. Precision = (# yes) / (sample size).

---

## Kill rule (claim FAIL if any of these hold)

- validation gate fails, OR
- prevalence among kernels marked correct < 5%, OR
- audit precision < 80%, OR
- control flag rate on PyTorch references > 1%.

No retuning of the detector after archive results are seen. Report the
exact counts either way.

---

## What will be reported (no estimates)

- verdict line: PASS or FAIL, and which condition decided it
- preregistration commit hash (this file)
- detector freeze commit hash
- total kernels scanned
- how many were `correct == True` (or equivalent)
- how many of those were nn.Module-style (stateful) vs purely functional
- prevalence among all correct kernels, and among the stateful subset
- per-level breakdown if levels exist
- control counts (references scanned, references flagged, rate)
- audit sample (seed 0) with yes/no + quoted caching line
- raw JSON of every flagged id
- 3 concrete example flagged kernels (ids + caching line) if any exist

Accuracy first. Never fake or estimate a number that was not measured.

---

## Constraints (also frozen)

- CPU only; no GPU rental; no CUDA execution needed.
- If the dataset cannot be downloaded or has a different schema, adapt to
  the real schema and say so. If it is not reachable at all, stop and
  report; do not invent archive numbers.
- Keep RAM and disk modest; stream the dataset if large.
