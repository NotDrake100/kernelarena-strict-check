"""Sanity checks for the K2 stateful eligibility helper (does not touch the detector)."""
import textwrap

from scripts.k2_stateful import is_stateful_kernel


def test_owns_linear_used_in_forward():
    src = textwrap.dedent(
        """
        import torch.nn as nn
        class ModelNew(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4)
            def forward(self, x):
                return self.linear(x)
        """
    )
    assert is_stateful_kernel(src)


def test_functional_matmul_not_stateful():
    src = textwrap.dedent(
        """
        import torch, torch.nn as nn
        class ModelNew(nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, A, B):
                return torch.matmul(A, B)
        """
    )
    assert not is_stateful_kernel(src)


def test_unused_parameter_not_stateful():
    src = textwrap.dedent(
        """
        import torch, torch.nn as nn
        class ModelNew(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.ones(2))
            def forward(self, x):
                return x
        """
    )
    assert not is_stateful_kernel(src)
