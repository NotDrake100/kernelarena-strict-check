# Pre-registration: K2 prevalence of stale-weight caching in stateful Python LLM kernels

**Status:** pre-registered. This file is committed *before* any candidate
corpus's solution bodies are inspected and *before* the frozen detector is
run on any K2 corpus. Dataset cards, schemas, and row counts may be read
*after* this commit solely to apply the selection rule below.

**Date:** 2026-10-18
**Repo:** kernelarena-strict-check (this repository)
**Experiment:** K2 (separate from K1)
**Method:** CPU-only static analysis of source text / Python AST (plus reading
embedded CUDA strings). No GPU rental. No CUDA execution of corpus kernels.

---

## Relation to K1 (do not edit K1)

K1 (`PREREG_stale_weight_prevalence.md` / `RESULTS_stale_weight_prevalence.md`)
scanned `SakanaAI/AI-CUDA-Engineer-Archive` and **FAILED**: 0/16459 kernels
marked `Correct` were flagged. That archive's `CUDA_Code` is C++
`forward(x, weight, …)` with weights passed every call, so the Python
instance-cache bug cannot occur there.

K1 stands. This file does not amend K1's claim, detector, thresholds, or
results. K2 asks a different question about a different population.

---

## Detector (frozen, unchanged)

The detector is `stale_weight_detector.py` frozen at commit

`f560e15571c1c49a47d177e4bd7a32c0540da28a`

**No detector change. No threshold change.** Before the scan, verify the
working-tree file is byte-identical to that commit (`git diff` empty and
matching SHA-256). If it is not identical, stop; do not retune.

---

## Question

In public corpora of LLM-written kernels that are **stateful Python
`nn.Module` solutions** (KernelBench-style `ModelNew` classes that own
weights, e.g. `load_inline` CUDA or Triton wrapped in a module), how
common is the stale-weight caching bug among kernels marked correct?

---

## Corpus selection rule (locked)

Apply this rule using **only** public dataset cards / README pages,
published schemas, and row counts / split sizes. Do **not** run the
detector and do **not** read solution bodies to choose. After the chosen
corpus is recorded in this experiment's working notes / results, it
cannot be switched.

### Eligibility of a corpus

A corpus is eligible iff all of the following hold from its card/schema:

1. **Public and free:** downloadable over HTTPS without login, payment, or
   a Hugging Face / GitHub gate (anonymous `curl` / `git clone` /
   Hugging Face resolve URL succeeds).
2. **LLM-generated:** the card or README states the kernel/solution field
   was produced by an LLM (or an LLM agent), not only human-written
   library kernels.
3. **Per-kernel correctness labels:** a boolean or equivalent per-row
   field (e.g. `correct`, `Correct`, `compiled_correct`,
   `correctness`, `passed`) distinguishing kernels that the publisher
   marked correct from those they did not.
4. **Python `nn.Module` solution source:** the solution field is Python
   source of a KernelBench-style `ModelNew` / `nn.Module` that owns
   weights (including modules that wrap `load_inline` CUDA or Triton).
   A corpus whose only generated field is a C++/CUDA `forward(x, weight,
   …)` extension (K1's archive) is **ineligible**.
5. **Not this repo's 16 KernelArena gate entries alone.** Those 16 are
   the validation set, not the prevalence corpus.

K1's archive is ineligible under (4). That follows from the committed K1
result and that archive's published schema (`CUDA_Code`), not from a new
read of its solution bodies.

### Proxy count used to pick among eligible corpora

Because solution bodies must not be read to choose, the ranking key is
the card/schema **proxy**:

> number of rows whose correctness label is true (or the publisher's
> equivalent) and whose solution field is described as Python
> `nn.Module` / `ModelNew` / `kernel.py` source.

If a card gives only a total row count and states that every row is a
correctness-labelled Python `ModelNew` solution, use the count of
correct-labelled rows if published; otherwise use the total row count
and record that the correct-subset size was not published.

### Search procedure (deterministic)

Search these sources, in this order, and keep every unique corpus that
appears:

1. Hugging Face dataset search, first 50 hits each, queries exactly:
   `kernelbench`, `kernel bench`, `ModelNew kernel`, `AI CUDA Engineer`,
   `triton llm kernel`, `kernelarena`.
2. GitHub code/repo search, first 20 hits each, queries exactly:
   `KernelBench ModelNew solutions dataset`,
   `LLM generated kernelbench correct`.
3. Named candidates already cited in this repository's docs/scripts
   (cards only): `SakanaAI/AI-CUDA-Engineer-Archive`,
   `wafer-ai/wafer-data`, `ScalingIntelligence/KernelBench`.

Deduplicate by canonical URL. For each unique hit, read only the dataset
card / README / schema / row-count metadata. Mark eligible or not with a
one-line reason. Rank eligible corpora by the proxy count (desc). Ties:
lexicographically smaller canonical URL.

**Pick the eligible corpus with the largest proxy count.**

If no eligible corpus exists, or the chosen corpus later yields fewer
than 200 *actually* stateful correct kernels under the eligibility rule
below, the claim **FAILS** (kill: no eligible corpus with ≥ 200 correct
stateful kernels).

Do not add a second corpus after seeing flags. Do not switch after this
file is committed and the rule has been applied.

---

## Eligibility rule for a 'stateful' kernel (locked)

A kernel source string is **stateful** iff a static Python AST check
finds:

1. a class that subclasses `nn.Module` / `Module` / `torch.nn.Module`
   (or a class named like a KernelBench `ModelNew` that defines both
   `__init__` and `forward` and calls `super().__init__`), **and**
2. that class's `__init__` creates or holds at least one parameter or
   buffer used by `forward`:
   - `self.<name> = nn.Parameter(...)` or `nn.Linear` / `Conv*` /
     `BatchNorm*` / `LayerNorm` / `Embedding` / other parameter-bearing
     `nn.*` constructors, or `register_parameter` / `register_buffer`,
   - and `forward` (or a method reachable from `forward`) reads that
     attribute or a parameter of that submodule (`weight`, `bias`,
     `running_mean`, `running_var`, …).

Purely functional sources (a lone `def forward(x, weight, …)`, raw
C++/CUDA with weights as arguments, or an `nn.Module` that owns no
parameters/buffers) are **not** stateful.

The frozen detector's `Detection.stateful` flag is a close subset of
this rule (it does not require the “used in forward” clause). K2's
denominator uses the rule in this section, implemented in the K2 scan
script **without editing** `stale_weight_detector.py`. Record both
counts if they differ.

Prevalence denominator = kernels that are **correct-labelled AND
stateful**. Flagged count = among those, `detect_stale_weight_cache(src).flagged`.

---

## Claim (frozen wording)

The detector frozen at `f560e15571c1c49a47d177e4bd7a32c0540da28a`
(unchanged; verified by hash/diff) flags **≥ 5%** of the **correct
stateful** kernels in the corpus selected by the rule above, and a
seeded (`random.Random(0)`, seed 0) manual audit of `min(30, n_flagged)`
flagged kernels finds **≥ 80%** are true stale-weight caching: a derived
tensor computed from weights and reused across calls without recompute
when weights change.

---

## What "stale-weight caching" means (same as K1)

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

---

## Control

Run the same frozen detector on the chosen corpus's PyTorch reference
models (the original non-generated `Model` / reference field). It must
flag **≤ 1%** of unique reference texts (SHA-256 of source).

If the corpus has no reference-model field, use this repository's
KernelBench reference files under `refs/` plus, if already present from
K1, the unique `PyTorch_Code_Module` texts recorded in
`results/stale_weight/control_refs.json` (229 unique KernelBench-style
refs). Do not invent references.

---

## Precision audit

1. Collect every flagged **correct stateful** kernel.
2. Sample with `random.Random(0).sample` of size `min(30, n_flagged)`
   (all of them if fewer than 30).
3. For each sampled kernel, read the code and decide yes/no: would an
   in-place weight update between calls produce a wrong output because a
   derived tensor is reused without recompute? Quote the offending line.
4. Precision = (# yes) / (sample size). Empty sample ⇒ precision
   undefined (not a < 80% fail by itself; prevalence kill still applies).

### Dynamic confirmation (if feasible on CPU)

Where a sampled flagged kernel is **pure PyTorch** (no compiled CUDA/HIP
extension, no Triton runtime required) **and** can be instantiated on
CPU from the published source plus the corpus's reference `get_inputs` /
shape metadata, run a cheap check:

- forward once; in-place `add_` / `copy_` on a parameter; forward again;
- compare against a freshly built module with the same updated weights.

Record pass/fail. If instantiation fails (missing CUDA, `load_inline`,
Triton, unknown shapes, import errors), say **blocked** and keep the
code-reading verdict. Never fabricate a dynamic result.

---

## Kill rule (claim FAIL if any of these hold)

- prevalence among correct stateful kernels < 5%, OR
- audit precision < 80%, OR
- control flag rate on references > 1%, OR
- no eligible corpus with ≥ 200 correct stateful kernels exists
  (including: the chosen corpus has fewer than 200 after the stateful
  rule is applied).

No retuning of the detector. No threshold change. No corpus switch after
this commit. Report the exact counts either way.

---

## What will be reported (no estimates)

- verdict line: PASS or FAIL, and which condition decided it
- this preregistration commit hash
- detector freeze commit hash and the hash/diff verification
- corpus chosen and why (proxy table of candidates)
- total kernels scanned
- how many were correct-labelled
- how many of those were stateful
- flagged among correct stateful; prevalence
- per-source / per-level breakdown if the corpus has those axes
- control counts
- audit table (seed 0) with yes/no + quoted line + dynamic result or
  “blocked / code-read only”
- raw JSON under `results/stale_weight_k2/`
- 3 concrete example flagged kernels (ids + offending lines) if any exist

Accuracy first. Never fake or estimate a number that was not measured.
If a step cannot run, report it as blocked.

---

## Constraints (also frozen)

- CPU only; no GPU rental; no CUDA execution required for the static scan.
- If the chosen dataset cannot be downloaded, stop and report; do not
  invent numbers and do not switch corpora.
- Keep RAM and disk modest; stream if large.
- Do not edit `PREREG_stale_weight_prevalence.md` or
  `RESULTS_stale_weight_prevalence.md`.
