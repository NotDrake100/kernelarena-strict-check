"""K2 eligibility: stateful Python nn.Module (does not edit the frozen detector).

A kernel is stateful iff a class subclasses nn.Module (or is a KernelBench-style
ModelNew with __init__+forward+super().__init__) AND that class's __init__
creates or holds at least one parameter/buffer that forward (or a method
reachable from forward) reads.
"""
from __future__ import annotations

import ast

_LAYER_CTORS = {
    "Parameter",
    "Linear",
    "Bilinear",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "LayerNorm",
    "GroupNorm",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "Embedding",
    "EmbeddingBag",
}
_PARAM_ATTRS = {
    "weight",
    "bias",
    "running_mean",
    "running_var",
    "running_std",
    "weight_g",
    "weight_v",
    "num_batches_tracked",
}


def is_stateful_kernel(src: str) -> bool:
    if not src or not str(src).strip():
        return False
    try:
        tree = ast.parse(str(src))
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and _is_module_class(node):
            if _class_holds_used_params(node):
                return True
    return False


def _is_module_class(cls: ast.ClassDef) -> bool:
    for b in cls.bases:
        name = _name_of(b)
        if name in {"Module", "nn.Module", "torch.nn.Module"} or name.endswith(".Module"):
            return True
    names = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
    if "forward" in names and "__init__" in names:
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        for n in ast.walk(init):
            if isinstance(n, ast.Call) and _name_of(n.func) in {"super", "super().__init__"}:
                return True
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                if isinstance(n.func.value, ast.Call) and _name_of(n.func.value.func) == "super":
                    return True
    return False


def _class_holds_used_params(cls: ast.ClassDef) -> bool:
    methods = {f.name: f for f in cls.body if isinstance(f, ast.FunctionDef)}
    init = methods.get("__init__")
    if init is None:
        return False
    layers, params, buffers = set(), set(), set()
    for node in ast.walk(init):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                attr = _self_attr(t)
                if not attr:
                    continue
                ctor = _ctor_name(node.value)
                if ctor in _LAYER_CTORS - {"Parameter"}:
                    layers.add(attr)
                elif ctor == "Parameter" or _is_parameter_ctor(node.value):
                    params.add(attr)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            buf = _register_name(node.value, "register_buffer")
            if buf:
                buffers.add(buf)
            par = _register_name(node.value, "register_parameter")
            if par:
                params.add(par)
    held = layers | params | buffers
    if not held:
        return False
    call_graph = {n: _called_self_methods(fn) for n, fn in methods.items()}
    reachable = _reachable("forward", call_graph) | _reachable("__call__", call_graph)
    for fname in reachable:
        fn = methods.get(fname)
        if fn is None:
            continue
        for n in ast.walk(fn):
            if not isinstance(n, ast.Attribute):
                continue
            if _is_self(n.value) and n.attr in held:
                return True
            if (
                isinstance(n.value, ast.Attribute)
                and _is_self(n.value.value)
                and n.value.attr in layers
                and n.attr in _PARAM_ATTRS
            ):
                return True
    return False


def _register_name(call: ast.Call, method: str) -> str | None:
    if not (
        isinstance(call.func, ast.Attribute)
        and _is_self(call.func.value)
        and call.func.attr == method
    ):
        return None
    if call.args and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str):
        return call.args[0].value
    return None


def _called_self_methods(fn: ast.FunctionDef) -> set[str]:
    out = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and _is_self(n.func.value):
            out.add(n.func.attr)
    return out


def _reachable(start: str, graph: dict) -> set[str]:
    seen, stack = set(), [start]
    while stack:
        n = stack.pop()
        if n in seen or n not in graph:
            continue
        seen.add(n)
        stack.extend(graph[n])
    return seen


def _is_self(n: ast.AST) -> bool:
    return isinstance(n, ast.Name) and n.id == "self"


def _self_attr(n: ast.AST) -> str | None:
    if isinstance(n, ast.Attribute) and _is_self(n.value):
        return n.attr
    return None


def _name_of(n: ast.AST) -> str:
    if isinstance(n, ast.Name):
        return n.id
    if isinstance(n, ast.Attribute):
        left = _name_of(n.value)
        return f"{left}.{n.attr}" if left else n.attr
    return ""


def _ctor_name(n: ast.AST) -> str | None:
    if not isinstance(n, ast.Call):
        return None
    name = _name_of(n.func)
    return name.split(".")[-1] if name else None


def _is_parameter_ctor(n: ast.AST) -> bool:
    return _ctor_name(n) == "Parameter"
