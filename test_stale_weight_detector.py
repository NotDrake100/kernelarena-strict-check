"""Unit tests + KernelArena labelled-set gate for stale_weight_detector.

Synthetic cases do not replace the real-label gate. The gate requires:
  all 6 known stale flagged, and at most 1 of the 10 known clean flagged.
"""
import os
import textwrap

from stale_weight_detector import detect_stale_weight_cache

# Labels recovered from results/gpu/*.json (see PREREG_stale_weight_prevalence.md).
STALE = [
    ("level2_73_Conv2d_BatchNorm_Scaling", "gemini_3_flash"),
    ("level2_98_Matmul_AvgPool_GELU_Scale_Max", "gemini_3_flash"),
    ("level2_98_Matmul_AvgPool_GELU_Scale_Max", "gpt_5.2"),
    ("level2_98_Matmul_AvgPool_GELU_Scale_Max", "gpt_5.2_min"),
    ("level2_98_Matmul_AvgPool_GELU_Scale_Max", "grok_4.1_min"),
    ("level4_104_FP8_Matmul", "opus_4.5"),
]
CLEAN = [
    ("level1_9_Tall_skinny_matrix_multiplication_", "gpt_5.2_min"),
    ("level2_40_Matmul_Scaling_ResidualAdd", "grok_4.1"),
    ("level2_40_Matmul_Scaling_ResidualAdd", "opus_4.5_min"),
    ("level2_55_Matmul_MaxPool_Sum_Scale", "grok_4.1_min"),
    ("level3_43_MinGPTCausalAttention", "grok_4.1_min"),
    ("level3_43_MinGPTCausalAttention", "opus_4.5"),
    ("level3_43_MinGPTCausalAttention", "opus_4.5_min"),
    ("level3_44_MiniGPTBlock", "gpt_5.2"),
    ("level4_104_FP8_Matmul", "gemini_3_flash"),
    ("level3_43_MinGPTCausalAttention", "kimi_k2"),
]
SOL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kb", "solutions")


def _det(src: str):
    return detect_stale_weight_cache(textwrap.dedent(src))


def test_synthetic_data_ptr_cache():
    r = _det(
        """
        import torch, torch.nn as nn
        class ModelNew(nn.Module):
            def __init__(self, K, N):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(K, N))
                self._c = None
                self._p = None
            def forward(self, x):
                if getattr(self, '_p', None) != self.weight.data_ptr():
                    self._c = self.weight.detach().clone()
                    self._p = self.weight.data_ptr()
                return x @ self._c
        """
    )
    assert r.flagged, r


def test_synthetic_id_cache():
    r = _det(
        """
        import torch, torch.nn as nn
        class ModelNew(nn.Module):
            def __init__(self, n):
                super().__init__()
                self.conv = nn.Conv2d(n, n, 3)
                self.fused_weight = None
                self.params_id = None
            def forward(self, x):
                current_id = id(self.conv.weight)
                if self.fused_weight is None or current_id != self.params_id:
                    self.fused_weight = self.conv.weight * 2
                    self.params_id = current_id
                return torch.nn.functional.conv2d(x, self.fused_weight)
        """
    )
    assert r.flagged, r


def test_synthetic_is_none_once():
    r = _det(
        """
        import torch, torch.nn as nn
        class ModelNew(nn.Module):
            def __init__(self, k, n):
                super().__init__()
                self.linear = nn.Linear(k, n)
                self.weight_pooled = None
            def forward(self, x):
                if self.weight_pooled is None:
                    self.weight_pooled = self.linear.weight.mean(0, keepdim=True)
                return x @ self.weight_pooled.t()
        """
    )
    assert r.flagged, r


def test_synthetic_init_register_buffer():
    r = _det(
        """
        import torch, torch.nn as nn
        class ModelNew(nn.Module):
            def __init__(self, k, n):
                super().__init__()
                self.linear = nn.Linear(k, n)
                self.register_buffer('Wgroup', self.linear.weight.view(2, -1, k).mean(1))
            def forward(self, x):
                return torch.nn.functional.linear(x, self.Wgroup)
        """
    )
    assert r.flagged, r


def test_synthetic_version_key_is_clean():
    r = _det(
        """
        import torch, torch.nn as nn
        class ModelNew(nn.Module):
            def __init__(self, k, n):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(k, n))
                self._c = None
                self._v = -1
            def forward(self, x):
                if self._c is None or self._v != self.weight._version:
                    self._c = self.weight.detach().clone()
                    self._v = self.weight._version
                return x @ self._c
        """
    )
    assert not r.flagged, r


def test_synthetic_always_recompute_is_clean():
    r = _det(
        """
        import torch, torch.nn as nn
        class ModelNew(nn.Module):
            def __init__(self, k, n):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(k, n))
            def forward(self, x):
                w_t = self.weight.t().contiguous()
                return x @ w_t
        """
    )
    assert not r.flagged, r


def test_synthetic_unused_cache_fields_are_clean():
    r = _det(
        """
        import torch, torch.nn as nn
        class ModelNew(nn.Module):
            def __init__(self, K, N):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(K, N))
                self.w_fp8 = None
                self.w_scale_inv = None
            def forward(self, x):
                w = (self.weight * 2).contiguous()
                return x @ w
        """
    )
    assert not r.flagged, r


def test_synthetic_causal_mask_buffer_is_clean():
    r = _det(
        """
        import torch, torch.nn as nn
        class ModelNew(nn.Module):
            def __init__(self, n, t):
                super().__init__()
                self.c_attn = nn.Linear(n, 3 * n)
                self.register_buffer('bias', torch.tril(torch.ones(t, t)))
            def forward(self, x):
                return self.c_attn(x) + self.bias[:x.size(1), :x.size(1)]
        """
    )
    assert not r.flagged, r


def test_synthetic_honest_linear_is_clean():
    r = _det(
        """
        import torch, torch.nn as nn
        class ModelNew(nn.Module):
            def __init__(self, k, n):
                super().__init__()
                self.linear = nn.Linear(k, n)
            def forward(self, x):
                return self.linear(x)
        """
    )
    assert not r.flagged and r.stateful, r


def test_synthetic_functional_matmul_not_stateful():
    r = _det(
        """
        import torch, torch.nn as nn
        class ModelNew(nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, A, B):
                return torch.matmul(A, B)
        """
    )
    assert not r.flagged and not r.stateful, r


def test_kernelarena_source_recoverable():
    missing = []
    for task, model in STALE + CLEAN:
        p = os.path.join(SOL, task, model, "kernel.py")
        if not os.path.exists(p):
            missing.append(p)
    assert not missing, (
        "Labelled KernelArena source is not recoverable. "
        "Run: bash scripts/fetch_solutions.sh kb. Missing: " + ", ".join(missing)
    )


def test_kernelarena_validation_gate():
    """Real-label gate: flag all 6 stale, at most 1 of 10 clean."""
    test_kernelarena_source_recoverable()
    stale_hits, clean_hits = [], []
    details = {}
    for task, model in STALE:
        src = open(os.path.join(SOL, task, model, "kernel.py")).read()
        d = detect_stale_weight_cache(src)
        details[f"{task}/{model}"] = d.as_dict()
        if d.flagged:
            stale_hits.append(f"{task}/{model}")
    for task, model in CLEAN:
        src = open(os.path.join(SOL, task, model, "kernel.py")).read()
        d = detect_stale_weight_cache(src)
        details[f"{task}/{model}"] = d.as_dict()
        if d.flagged:
            clean_hits.append(f"{task}/{model}")
    assert len(stale_hits) == 6, (
        f"validation gate: flagged {len(stale_hits)}/6 stale: {stale_hits}\n{details}"
    )
    assert len(clean_hits) <= 1, (
        f"validation gate: flagged {len(clean_hits)}/10 clean: {clean_hits}\n"
        + "\n".join(f"{k}: {details[k]}" for k in clean_hits)
    )
