"""Stricter KernelBench-style correctness checker (pure PyTorch, device-agnostic).
Checks beyond a single torch.allclose:
  1. fresh random inputs on every call (catches output/input caching)
  2. weights changed IN PLACE between calls, same data_ptr (catches stale weight caches)
  3. several seeds
  4. edge-case shapes (via a user-supplied input generator per shape)
  5. input-mutation and output-aliasing checks
plus static_audit(): custom-kernel markers vs PyTorch ops called in ModelNew.forward."""
import ast,re,copy,torch
from dataclasses import dataclass,field

@dataclass
class CheckResult:
    passed: bool
    failures: list=field(default_factory=list)
    n_checks: int=0
    max_abs_errs: list=field(default_factory=list)
    n_param_pairs: int=0
    n_new_params: int=0

def _cmp(a,b,rtol,atol):
    if isinstance(a,(tuple,list)): return all(_cmp(x,y,rtol,atol) for x,y in zip(a,b)) and len(a)==len(b)
    return a.shape==b.shape and torch.allclose(a.float(),b.float(),rtol=rtol,atol=atol,equal_nan=True)

def _pairs(ref,new,param_map=None):
    """(ref_param,new_param) pairs: by explicit new->ref name map, else identical names."""
    rs=dict(ref.named_parameters()); m=param_map or {}
    return [(rs[m.get(k,k)],p) for k,p in new.named_parameters() if m.get(k,k) in rs and rs[m.get(k,k)].shape==p.shape]

def strict_check(ref_cls,new_cls,init_args,make_inputs,shapes,seeds=(0,1,2),calls_per_seed=3,
                 rtol=1e-3,atol=1e-3,perturb_weights=True,device='cpu',param_map=None):
    """make_inputs(shape, generator)->list[Tensor]. Each shape key is passed to make_inputs."""
    res=CheckResult(True)
    def fail(msg): res.passed=False; res.failures.append(msg)
    for seed in seeds:
        torch.manual_seed(seed); ref=ref_cls(*init_args).to(device).eval()
        torch.manual_seed(seed); new=new_cls(*init_args).to(device).eval()   # KernelBench-style identical seeding
        pairs=_pairs(ref,new,param_map); res.n_param_pairs=len(pairs); res.n_new_params=len(list(new.parameters()))
        with torch.no_grad():
            for pr,pn in pairs: pn.copy_(pr)
        g=torch.Generator(device=device).manual_seed(1000+seed)
        for shape in shapes:
            for c in range(calls_per_seed):
                if perturb_weights and c>0:
                    with torch.no_grad():   # in-place: data_ptr unchanged
                        for pr,pn in pairs:
                                d=torch.randn(pr.shape,generator=g,device=device).to(pr)*pr.detach().float().std().clamp(min=1e-3).to(pr)
                                pr.add_(d); pn.add_(d)
                inp=[t.to(device) for t in make_inputs(shape,g)]
                snap=[t.clone() for t in inp]
                with torch.no_grad():
                    try: y_new=new(*inp)
                    except Exception as e: fail(f'seed{seed} shape{shape} call{c}: exception {type(e).__name__}: {e}'); continue
                    mutated=any(not torch.equal(a,b) for a,b in zip(inp,snap))
                    if mutated: fail(f'seed{seed} shape{shape} call{c}: ModelNew mutated its inputs')
                    y_ref=ref(*snap)
                res.n_checks+=1
                try: res.max_abs_errs.append(float((y_new.float()-y_ref.float()).abs().max()))
                except Exception: res.max_abs_errs.append(None)
                if isinstance(y_new,torch.Tensor) and any(y_new.data_ptr()==t.data_ptr() for t in inp if t.numel()):
                    fail(f'seed{seed} shape{shape} call{c}: output aliases an input')
                if not _cmp(y_new,y_ref,rtol,atol):
                    err=(y_new.float()-y_ref.float()).abs().max().item() if isinstance(y_new,torch.Tensor) and y_new.shape==y_ref.shape else 'shape mismatch'
                    fail(f'seed{seed} shape{shape} call{c}: mismatch max_abs_err={err}')
    return res

KERNEL_MARKERS=re.compile(r'__global__|@triton\.jit|tl\.load|load_inline|hipblas|rocblas|composable_kernel|ck::|asm\s*\(')
TORCH_CALL=re.compile(r'\b(?:torch\.(?:nn\.functional\.)?|F\.)([a-zA-Z_]\w*)\s*\(')
TRIVIAL={'empty','empty_like','zeros','zeros_like','ones','ones_like','tensor','arange','cat','stack','no_grad','device','float32','float16','is_tensor'}
def static_audit(src:str)->dict:
    out={'has_custom_kernel_marker':bool(KERNEL_MARKERS.search(src)),'forward_found':False,'forward_torch_ops':[],
         'forward_submodule_calls':[],'custom_ext_calls_in_forward':0,'fallback_branch':bool(re.search(r'USE_\w*\s*=\s*False|except\s+Exception',src))}
    try: tree=ast.parse(src)
    except SyntaxError as e: out['syntax_error']=str(e); out['verdict']='syntax_error'; return out
    for n in ast.walk(tree):
        if isinstance(n,ast.ClassDef) and n.name=='ModelNew':
            for f in n.body:
                if isinstance(f,ast.FunctionDef) and f.name=='forward':
                    out['forward_found']=True; fs=ast.get_source_segment(src,f)
                    out['forward_torch_ops']=sorted({m for m in TORCH_CALL.findall(fs) if m not in TRIVIAL})
                    out['forward_submodule_calls']=sorted(set(re.findall(r'self\.(\w+)\s*\(',fs)))
                    out['custom_ext_calls_in_forward']=len(re.findall(r'\b\w*(?:_ext|_module|_cuda|_hip|kernel)\w*\.\w+\s*\(',fs))
    out['verdict']=('no_custom_kernel' if not out['has_custom_kernel_marker'] else
                    'custom_kernel_unused_in_forward' if out['forward_found'] and out['custom_ext_calls_in_forward']==0 else 'custom_kernel_used')
    return out
