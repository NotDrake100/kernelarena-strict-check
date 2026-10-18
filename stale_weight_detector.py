"""Static detector for stale-weight caching.

A module caches a derived copy of a live parameter/buffer, built once or keyed
on data_ptr()/id()/shape/device/dtype/is-None/a boolean ready-flag, and reuses
it on later forward calls without checking the parameter's _version or values.

CPU-only: Python AST plus a conservative scan of embedded/raw CUDA strings.
This module is the frozen detector once committed; do not retune after the
archive scan.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

# Live parameter-bearing nn.* constructors (and aliases).
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

# Attributes that are parameters or fused-into-weight buffers on those layers.
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

_WEAK_FUNCS = {"data_ptr", "id", "storage_offset", "storage"}
_WEAK_ATTRS = {"device", "dtype", "shape", "ndim", "is_cuda"}
_STRONG_ATTRS = {"_version"}

# C++: persistent weight storage keyed on pointer / null, not per-call args.
_CPP_STATIC_CACHE = re.compile(
    r"\bstatic\s+(?:torch::Tensor|at::Tensor|void\*|float\*|half\*|__half\*)\s+"
    r"(\w*(?:cache|cached|weight|w_t|w_pack|fused|quant|dequant|packed|precomp)\w*)",
    re.IGNORECASE,
)
_CPP_PTR_KEY = re.compile(
    r"(?:data_ptr|data\(\)|storage\(\)|get\(\))\s*(?:<[^>]+>)?\s*\(\s*\)"
    r".*(?:==|!=).*(?:nullptr|NULL|last_|prev_|cached_|saved_)"
    r"|(?:nullptr|NULL|last_|prev_|cached_|saved_).*(?:==|!=).*(?:data_ptr|data\(\))",
    re.IGNORECASE | re.DOTALL,
)
_CPP_ONCE_FLAG = re.compile(
    r"\bstatic\s+(?:bool|int|void\*)\s+\w*(?:init|ready|cached|copied|valid)\w*\s*(?:=\s*(?:false|0|nullptr))?",
    re.IGNORECASE,
)


@dataclass
class Finding:
    attr: str
    reason: str
    line: int
    snippet: str
    keys: list[str] = field(default_factory=list)


@dataclass
class Detection:
    flagged: bool
    findings: list[Finding] = field(default_factory=list)
    stateful: bool = False
    parse_kind: str = "python"  # python | cuda | mixed | empty | syntax_error
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "flagged": self.flagged,
            "stateful": self.stateful,
            "parse_kind": self.parse_kind,
            "findings": [
                {
                    "attr": f.attr,
                    "reason": f.reason,
                    "line": f.line,
                    "snippet": f.snippet,
                    "keys": f.keys,
                }
                for f in self.findings
            ],
            "notes": self.notes,
        }


def detect_stale_weight_cache(src: str) -> Detection:
    """Return whether `src` (Python and/or CUDA text) has stale-weight caching."""
    if src is None or not str(src).strip():
        return Detection(False, parse_kind="empty")
    src = str(src)
    py_det = _detect_python(src)
    cu_findings = _detect_cuda(src)
    kind = py_det.parse_kind
    if cu_findings and kind == "syntax_error":
        kind = "cuda"
    elif cu_findings and kind == "python":
        kind = "mixed"
    findings = list(py_det.findings) + cu_findings
    return Detection(
        flagged=bool(findings),
        findings=findings,
        stateful=py_det.stateful,
        parse_kind=kind,
        notes=py_det.notes,
    )


def is_stateful_module(src: str) -> bool:
    """nn.Module-style (holds parameters / layer submodules) vs purely functional."""
    return detect_stale_weight_cache(src).stateful


# ---------------------------------------------------------------------------
# Python AST
# ---------------------------------------------------------------------------

def _detect_python(src: str) -> Detection:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        # Raw CUDA / C++ will fail parse; still try to see if a Python class
        # is embedded (rare). Otherwise this is not Python.
        return Detection(False, parse_kind="syntax_error")
    stateful = False
    findings: list[Finding] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            c = _analyse_class(node, src)
            stateful = stateful or c["stateful"]
            findings.extend(c["findings"])
    return Detection(bool(findings), findings, stateful=stateful, parse_kind="python")


def _analyse_class(cls: ast.ClassDef, src: str) -> dict:
    if not _is_module_class(cls):
        return {"stateful": False, "findings": []}
    methods = {f.name: f for f in cls.body if isinstance(f, ast.FunctionDef)}
    layers, params, buffers = _collect_params(methods.get("__init__"))
    stateful = bool(layers or params)
    if not stateful:
        return {"stateful": False, "findings": []}

    writes = []  # (attr, func_name, node, keys)
    for fname, fn in methods.items():
        tensor_names: set[str] = set()
        ptrkey_names: set[str] = set()
        for stmt in ast.walk(fn):
            if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                value = stmt.value if isinstance(stmt, ast.AnnAssign) else stmt.value
                if value is None:
                    continue
                targets = [stmt.target] if isinstance(stmt, ast.AnnAssign) else stmt.targets
                meta = _is_metadata_expr(value) or (
                    isinstance(value, ast.Name) and value.id in ptrkey_names
                )
                rhs_tainted = (not meta) and _expr_tainted(
                    value, layers, params, tensor_names
                )
                if _is_metadata_expr(value) or _expr_is_data_ptr(value):
                    for t in targets:
                        if isinstance(t, ast.Name):
                            ptrkey_names.add(t.id)
                if rhs_tainted:
                    for t in targets:
                        if isinstance(t, ast.Name):
                            tensor_names.add(t.id)
                        attr = _self_attr(t)
                        if attr and attr not in layers and attr not in params:
                            if _is_parameter_ctor(value):
                                params.add(attr)
                                continue
                            keys = _enclosing_keys(stmt, fn, ptrkey_names, methods)
                            writes.append((attr, fname, stmt, keys))
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                buf = _register_buffer_write(stmt.value)
                if buf:
                    name, val = buf
                    if _expr_tainted(val, layers, params, tensor_names) and not _is_metadata_expr(val):
                        keys = _enclosing_keys(stmt, fn, ptrkey_names, methods)
                        writes.append((name, fname, stmt, keys))

    call_graph = {n: _called_self_methods(fn) for n, fn in methods.items()}
    reachable = _reachable("forward", call_graph) | _reachable("__call__", call_graph)

    # Attributes read in reachable methods (loads, not the store itself).
    reads = _attr_reads(methods, reachable)

    findings = []
    seen = set()
    for attr, fname, node, keys in writes:
        if attr in layers or attr in params:
            continue
        if attr not in reads:
            continue
        effective = list(keys)
        if not effective:
            if fname == "__init__":
                effective = ["built_once"]
            elif fname in {"forward", "__call__"}:
                continue  # unconditional refresh on every call
            else:
                effective = _callsite_keys(fname, methods, call_graph)
                if not effective:
                    # Helper is always invoked from forward: refreshes every call.
                    if fname in reachable:
                        continue
                    effective = ["built_once"]
        if _has_strong_key(effective):
            continue
        if not _is_weak_only(effective):
            continue
        key = (attr, getattr(node, "lineno", 0))
        if key in seen:
            continue
        seen.add(key)
        snippet = _line(src, getattr(node, "lineno", 1))
        findings.append(
            Finding(
                attr=attr,
                reason=_reason(attr, effective, fname),
                line=getattr(node, "lineno", 0),
                snippet=snippet,
                keys=sorted(set(effective)),
            )
        )
    return {"stateful": True, "findings": findings}


def _is_module_class(cls: ast.ClassDef) -> bool:
    for b in cls.bases:
        name = _name_of(b)
        if name in {"Module", "nn.Module", "torch.nn.Module"} or name.endswith(".Module"):
            return True
    # Fallback: any class that defines forward and __init__ with super().__init__
    names = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
    return "forward" in names and "__init__" in names


def _collect_params(init: ast.FunctionDef | None):
    layers, params, buffers = set(), set(), set()
    if init is None:
        return layers, params, buffers
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
            buf = _register_buffer_write(node.value)
            if buf:
                buffers.add(buf[0])
            if _is_self_method_call(node.value, "register_parameter"):
                args = node.value.args
                if len(args) >= 1 and isinstance(args[0], ast.Constant) and isinstance(args[0].value, str):
                    params.add(args[0].value)
    return layers, params, buffers


def _expr_tainted(expr: ast.AST, layers: set, params: set, local_taint: set) -> bool:
    if expr is None:
        return False
    if _is_parameter_ctor(expr):
        return False
    for n in ast.walk(expr):
        if isinstance(n, ast.Name) and n.id in local_taint:
            return True
        if isinstance(n, ast.Attribute):
            if isinstance(n.value, ast.Attribute) and _is_self(n.value.value):
                # self.linear.weight
                if n.value.attr in layers and n.attr in _PARAM_ATTRS:
                    return True
            if _is_self(n.value) and n.attr in params:
                return True
            # self.weight when it is a Parameter on this module
            if _is_self(n.value) and n.attr in _PARAM_ATTRS and n.attr in params:
                return True
        if isinstance(n, ast.Name) and n.id in _PARAM_ATTRS:
            # local alias `weight` only counts if already tainted via assignment
            pass
    return False


def _enclosing_keys(
    stmt: ast.AST,
    fn: ast.FunctionDef,
    ptrkey_names: set[str] | None = None,
    methods: dict | None = None,
) -> list[str]:
    """Weak/strong cache keys from `if` tests that enclose `stmt`."""
    keys: list[str] = []
    parent_ifs = _enclosing_ifs(stmt, fn)
    for test in parent_ifs:
        keys.extend(_keys_from_test(test, fn, ptrkey_names or set(), methods or {}))
    return keys


def _enclosing_ifs(stmt: ast.AST, fn: ast.FunctionDef) -> list[ast.AST]:
    tests = []
    for node in ast.walk(fn):
        if isinstance(node, ast.If) and _stmt_in(stmt, node.body):
            tests.append(node.test)
    return tests


def _stmt_in(stmt: ast.AST, body: list) -> bool:
    sl = getattr(stmt, "lineno", None)
    if sl is None:
        return False
    for b in body:
        if b is stmt:
            return True
        if getattr(b, "lineno", 0) <= sl <= getattr(b, "end_lineno", getattr(b, "lineno", 0)):
            # stmt lives inside this body node's subtree
            for c in ast.walk(b):
                if c is stmt:
                    return True
    return False


def _keys_from_test(
    test: ast.AST, fn: ast.FunctionDef, ptrkey_names: set[str], methods: dict
) -> list[str]:
    keys = []
    for n in ast.walk(test):
        if isinstance(n, ast.Call):
            nm = _name_of(n.func)
            if nm in _WEAK_FUNCS or nm.endswith(".data_ptr"):
                keys.append(
                    "data_ptr"
                    if "data_ptr" in nm or nm == "data_ptr"
                    else nm.split(".")[-1]
                )
            if nm == "id" or nm.endswith(".id"):
                keys.append("id")
            # self._get_params_id() etc.: inspect callee for id()/data_ptr()
            if isinstance(n.func, ast.Attribute) and _is_self(n.func.value):
                keys.extend(_keys_from_helper(n.func.attr, methods))
        if isinstance(n, ast.Attribute) and n.attr in _WEAK_ATTRS:
            keys.append(n.attr)
        if isinstance(n, ast.Attribute) and n.attr in _STRONG_ATTRS:
            keys.append("_version")
        if isinstance(n, ast.Attribute) and n.attr in {"data_ptr"}:
            keys.append("data_ptr")
        if isinstance(n, ast.Name) and n.id in ptrkey_names:
            keys.append("data_ptr")
        if isinstance(n, ast.Compare):
            if any(isinstance(op, (ast.Is, ast.IsNot)) for op in n.ops):
                for comp in list(n.comparators) + [n.left]:
                    if isinstance(comp, ast.Constant) and comp.value is None:
                        keys.append("is_none")
        if isinstance(n, ast.Attribute) and _is_self(n.value):
            if n.attr.endswith(
                ("_ready", "_initialized", "_init", "_valid", "_built", "_cached")
            ):
                keys.append("ready_flag")
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
            inner = n.operand
            if isinstance(inner, ast.Attribute) and _is_self(inner.value):
                keys.append("ready_flag")
            if isinstance(inner, ast.Name):
                keys.append("ready_flag")
    return keys


def _keys_from_helper(name: str, methods: dict) -> list[str]:
    fn = methods.get(name)
    if fn is None:
        return []
    keys = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Call):
            nm = _name_of(n.func)
            if nm == "id" or nm.endswith(".id"):
                keys.append("id")
            if "data_ptr" in nm:
                keys.append("data_ptr")
    return keys


def _has_strong_key(keys: list[str]) -> bool:
    return any(k in {"_version", "values"} for k in keys)


def _is_weak_only(keys: list[str]) -> bool:
    allowed = {
        "data_ptr",
        "id",
        "is_none",
        "device",
        "dtype",
        "shape",
        "size",
        "ndim",
        "is_cuda",
        "ready_flag",
        "built_once",
        "storage_offset",
        "storage",
    }
    return bool(keys) and all(k in allowed for k in keys)


def _callsite_keys(helper: str, methods: dict, call_graph: dict) -> list[str]:
    """Keys of `if` tests that wrap calls to `helper`."""
    keys = []
    for fname, fn in methods.items():
        if helper not in call_graph.get(fname, set()):
            continue
        ptrkey_names = _ptrkey_names_in(fn)
        for node in ast.walk(fn):
            if isinstance(node, ast.Call) and _is_self_method_call(node, helper):
                keys.extend(_enclosing_keys(node, fn, ptrkey_names, methods))
    return keys


def _ptrkey_names_in(fn: ast.FunctionDef) -> set[str]:
    names = set()
    for stmt in ast.walk(fn):
        if isinstance(stmt, ast.Assign) and _expr_is_data_ptr(stmt.value):
            for t in stmt.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
    return names


def _is_metadata_expr(expr: ast.AST) -> bool:
    """True if expr is identity/layout metadata, not a derived tensor copy."""
    if isinstance(expr, ast.Attribute) and expr.attr in _WEAK_ATTRS | {"data_ptr", "id"}:
        return True
    return _expr_is_data_ptr(expr)


def _expr_is_data_ptr(expr: ast.AST) -> bool:
    if isinstance(expr, ast.Call):
        nm = _name_of(expr.func)
        if nm in {"data_ptr", "id"} or nm.endswith(".data_ptr") or nm.endswith(".id"):
            return True
    return False


def _called_self_methods(fn: ast.FunctionDef) -> set[str]:
    out = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and _is_self(n.func.value):
            out.add(n.func.attr)
    return out


def _reachable(start: str, graph: dict) -> set[str]:
    seen = set()
    stack = [start]
    while stack:
        n = stack.pop()
        if n in seen or n not in graph:
            continue
        seen.add(n)
        stack.extend(graph[n])
    return seen


def _attr_reads(methods: dict, reachable: set) -> set[str]:
    reads = set()
    for fname in reachable:
        fn = methods.get(fname)
        if fn is None:
            continue
        for n in ast.walk(fn):
            if isinstance(n, ast.Attribute) and _is_self(n.value) and isinstance(n.ctx, ast.Load):
                reads.add(n.attr)
    return reads


def _reason(attr: str, keys: list[str], fname: str) -> str:
    k = ",".join(sorted(set(keys)))
    where = "in __init__" if fname == "__init__" else f"in {fname}()"
    return f"caches derived parameter copy on self.{attr} {where}; rebuild keys=[{k}] (no _version/values)"


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


def _is_self_method_call(n: ast.AST, method: str) -> bool:
    return (
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and _is_self(n.func.value)
        and n.func.attr == method
    )


def _register_buffer_write(call: ast.Call):
    if not (
        isinstance(call.func, ast.Attribute)
        and _is_self(call.func.value)
        and call.func.attr == "register_buffer"
    ):
        return None
    if not call.args:
        return None
    name = None
    if isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str):
        name = call.args[0].value
    if name is None or len(call.args) < 2:
        return None
    return name, call.args[1]


def _line(src: str, lineno: int) -> str:
    lines = src.splitlines()
    if 1 <= lineno <= len(lines):
        return lines[lineno - 1].strip()[:240]
    return ""


# ---------------------------------------------------------------------------
# CUDA / C++ (conservative: only pointer-keyed or once-flagged static caches)
# ---------------------------------------------------------------------------

def _detect_cuda(src: str) -> list[Finding]:
    findings = []
    if "__global__" not in src and "torch::Tensor" not in src and "cudaMemcpy" not in src:
        return findings
    # Static persistent tensors/pointers whose *name* says they cache weights,
    # AND a pointer/null key or a once-flag appears in the same file.
    statics = list(_CPP_STATIC_CACHE.finditer(src))
    has_ptr_key = bool(_CPP_PTR_KEY.search(src))
    has_once = bool(_CPP_ONCE_FLAG.search(src))
    if not statics:
        return findings
    if not (has_ptr_key or has_once):
        return findings
    for m in statics:
        line = src[: m.start()].count("\n") + 1
        findings.append(
            Finding(
                attr=m.group(1),
                reason=(
                    f"C++ static weight cache `{m.group(1)}` rebuilt on "
                    f"{'data_ptr/null' if has_ptr_key else 'once-flag'}, not values/_version"
                ),
                line=line,
                snippet=m.group(0)[:240],
                keys=["data_ptr" if has_ptr_key else "built_once"],
            )
        )
    return findings
