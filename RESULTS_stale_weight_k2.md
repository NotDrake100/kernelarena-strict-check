# Results: K2 prevalence of stale-weight caching in stateful Python LLM kernels

**Verdict: FAIL (prevalence < 5%).**

The frozen detector flagged **0 / 1684** kernels that were both marked `correct`
and stateful (prevalence **0.0%**). The pre-registered claim required ≥ 5%.
Other kill-rule conditions did not fail:

| Condition | Result | Threshold |
|---|---|---|
| Eligible corpus with ≥ 200 correct stateful kernels | PASS: 1684 | ≥ 200 |
| Prevalence among correct stateful | **FAIL: 0/1684 = 0.0%** | ≥ 5% |
| Audit precision (seed 0) | N/A: 0 flagged, sample empty | ≥ 80% of sampled flags |
| Control on PyTorch references | PASS: 0/236 = 0.0% | ≤ 1% |

No detector change. No threshold change. No corpus switch after the prereg
commit. Every count below was measured. K1
(`RESULTS_stale_weight_prevalence.md`) is not edited.

---

## Frozen hashes

| Item | Git commit / digest |
|---|---|
| K2 pre-registration (`PREREG_stale_weight_k2.md`) | `02df4269a5535523ebf39c62a131d4d099888b5f` |
| Detector freeze (K1, unchanged) | `f560e15571c1c49a47d177e4bd7a32c0540da28a` |
| `stale_weight_detector.py` SHA-256 (working = freeze) | `ba5abbdda608f99db3d30abcc714a36eef376a3f9116d81ababedf8f60aa8ec0` |
| `git diff f560e15 -- stale_weight_detector.py` | empty |

Freeze check is also in `results/stale_weight_k2/scan_summary.json`.

---

## Corpus chosen and why

**Chosen:** Hugging Face dataset `ScalingIntelligence/kernelbench-samples`
(git clone, no login).

Selection used only cards, schemas, and row/label counts, after the prereg
commit, with no detector run and no solution-body read to choose. Ranking
key = number of rows with a true correctness label and a Python `nn.Module`
/ `ModelNew` / `kernel.py` solution field.

| Corpus | Eligible? | Proxy (correct-labelled Python modules) |
|---|---|---:|
| `ScalingIntelligence/kernelbench-samples` | yes | **8157** (`correct==True` on 49599 `kernel.json` rows) |
| `Infatoshi/kernelbench-v3-runs` | yes | 927 (`correct` in `results.csv` with `solution_link`) |
| `wafer-ai/wafer-data` `kernelbench-hip-mi300x` | yes | 417 (`passed==true` in `index.json`) |
| `Infatoshi/kernelbench-hard-submissions` | yes | ≤ 53 |
| `SakanaAI/AI-CUDA-Engineer-Archive` | no | C++ `CUDA_Code` (K1) |
| `ScalingIntelligence/KernelBench` | no | problems, not LLM solutions |
| traces / prompts / mutation artifacts | no | see `results/stale_weight_k2/corpus_selection.json` |

`kernel.json` schema keys (label/metadata only): `hardware`, `level`,
`problem_id`, `problem_name`, `run_name`, `model_name`, `kernel`, `correct`,
`eval_result`. Field used for correctness: **`correct`**. Field used for
source: **`kernel`** (Python `ModelNew` / `load_inline` text).

iterative_refinement files are `log.json` eval logs, not the same labelled
`kernel.json` rows; they were not added after seeing flags (there were none).

---

## Schema / eligibility actually used

Stateful = K2 rule in `scripts/k2_stateful.py`: an `nn.Module` subclass
whose `__init__` creates or holds parameters/buffers that `forward` (or a
method reachable from `forward`) reads. The frozen detector's
`Detection.stateful` agreed on every correct row: **1684 = 1684**.

Most level-1 correct kernels take activation tensors as arguments and own
no weights (functional). Level 2/3 `ModelNew` classes usually own
`nn.Linear` / `Conv*` / `BatchNorm*` and are stateful.

---

## Corpus counts (measured)

| Quantity | Count |
|---|---:|
| Total `kernel.json` scanned | 49599 |
| `correct == True` | 8157 |
| Correct and stateful (K2 rule) | 1684 |
| Correct and functional | 6473 |
| Flagged among correct (any) | 0 |
| Flagged among correct stateful | 0 |
| Prevalence among correct stateful | 0 / 1684 = **0.0%** |
| Correct with empty `kernel` | 0 |
| Correct that failed `ast.parse` | 0 |

### Per-split breakdown

| Split | Scanned | Correct | Stateful (correct) | Functional (correct) | Flagged (correct stateful) |
|---|---:|---:|---:|---:|---:|
| baseline_eval | 1724 | 529 | 223 | 306 | 0 |
| repeated_sampling | 47875 | 7628 | 1461 | 6167 | 0 |

### Per-level breakdown

| Level | Scanned | Correct | Stateful (correct) | Functional (correct) | Flagged (correct stateful) |
|---:|---:|---:|---:|---:|---:|
| 1 | 20081 | 6314 | 124 | 6190 | 0 |
| 2 | 19290 | 669 | 666 | 3 | 0 |
| 3 | 10228 | 1174 | 894 | 280 | 0 |

### Per-model breakdown (correct stateful)

| model_name | Scanned | Correct | Stateful (correct) | Flagged |
|---|---:|---:|---:|---:|
| deepseek-chat | 24583 | 5695 | 1352 | 0 |
| meta-llama/Llama-3.1-70B-Instruct | 18612 | 1965 | 105 | 0 |
| o1-2024-12-17 | 239 | 139 | 85 | 0 |
| deepseek-reasoner | 246 | 133 | 83 | 0 |
| (empty `model_name`; path `llama-3.1-70b`) | 4927 | 32 | 20 | 0 |
| claude-3-5-sonnet-20241022 | 246 | 69 | 19 | 0 |
| gpt-4o-2024-08-06 | 247 | 55 | 16 | 0 |
| Meta-Llama-3.1-405B-Instruct | 250 | 42 | 4 | 0 |
| meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo | 249 | 27 | 0 | 0 |

---

## Control (PyTorch references)

The samples corpus has no reference-model field. Per the prereg, control is
the unique K1 `PyTorch_Code_Module` texts (229 hashes, source not stored in
the K1 dump) plus this repository's `refs/*.py` re-detected with the frozen
detector. Seven local ref files are extra hashes, for **236** unique texts.

| Quantity | Count |
|---|---:|
| Unique references | 236 |
| Flagged | 0 |
| Rate | 0 / 236 = **0.0%** (≤ 1%, control passes) |

Raw dump: `results/stale_weight_k2/control_refs.json`.

---

## Precision audit (seed 0)

`random.Random(0)` was prepared to sample 30 flagged ids, or all flagged if
fewer than 30.

| Quantity | Value |
|---|---|
| Flagged pool (correct stateful) | 0 |
| Sample size | 0 (all of them) |
| Yes (true stale-weight caching) | 0 |
| Precision | **undefined** (empty sample; not a < 80% fail) |
| Dynamic confirmation | not run (nothing to instantiate) |

Audit method would have been code-reading. CPU dynamic confirmation of
`load_inline` / CUDA / Triton `ModelNew` classes is blocked on this VM
(no GPU, no compiled extensions). That did not arise: the flag list is empty.

Raw dump: `results/stale_weight_k2/audit_sample.json`.
Flagged ids: `results/stale_weight_k2/flagged_kernels.json` (empty list).

### Three example flagged kernels

None exist. The frozen detector produced an empty flag list on this corpus.

---

## What was inspected after the scan (not used to retune)

Token counts among the 1684 correct stateful kernels (after the scan):

| Token | Hits |
|---|---:|
| `data_ptr` | 1682 |
| `id(` | 125 |
| `register_buffer` | 84 |
| `self._cache` / `fused_weight` / `cached` | 0 / 0 / 0 |

Spot checks of `data_ptr` lines are CUDA launch arguments
(`weight.data_ptr<float>()`), not pointer-keyed Python caches. Spot checks
of `register_buffer` are BatchNorm running stats and causal-mask `torch.tril`
buffers, not derived copies of live parameters.

These are why a looser heuristic would have been tempting and why the freeze
rule forbids retuning.

---

## Kill-rule application

```
FAIL because prevalence 0.0% < 5%.
```

The detector is not too loose on the control (0.0% ≤ 1%). An eligible corpus
with 1684 correct stateful Python `ModelNew` kernels exists, so the “no
corpus ≥ 200” kill does not apply. On this 2024–2025 KernelBench samples
release, the instance-cache pattern the detector was frozen on (KernelArena
2026 HIP/CUDA `ModelNew` modules) does not appear in kernels marked correct.

That is a failed claim, not a failed measurement.
