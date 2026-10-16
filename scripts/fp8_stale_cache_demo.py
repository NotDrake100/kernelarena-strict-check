import os,sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
"""Runs the REAL ModelNew from level4_104_FP8_Matmul/opus_4.5/kernel.py on CPU.
On CPU the HIP extension fails to compile, so the file's own PyTorch fallback path is used for
input quantization; the weight-cache logic (_maybe_update_weight_cache) is the unmodified original.
Reference = the same FP8 quant-dequant applied to the CURRENT weight on every call (assumed
reference semantics; the upstream reference file is not in wafer-data)."""
import importlib.util,json,torch,sys,io,contextlib
P=os.environ.get('WAFER_DATA','wafer-data')+'/kernel-arena/exports/kernelbench-hip-mi300x/solutions/level4_104_FP8_Matmul/opus_4.5/kernel.py'
buf=io.StringIO()
with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
    spec=importlib.util.spec_from_file_location('k',P);k=importlib.util.module_from_spec(spec);spec.loader.exec_module(k)
def ref(x,w,fmax=448.0,dt=torch.float8_e4m3fn):
    def qd(t):
        s=fmax/t.abs().max().clamp(min=1e-12); return ((t*s).clamp(-fmax,fmax).to(dt).to(t.dtype))/s
    x2=x.reshape(-1,w.shape[0]); return (qd(x2)@qd(w)).reshape(x.shape[0],x.shape[1],w.shape[1])
out={'use_custom_kernels':k.USE_CUSTOM_KERNELS,'compile_log_head':buf.getvalue()[:300],'runs':[]}
for seed in [0,1,2]:
    torch.manual_seed(seed)
    B,T,K,N=2,64,256,128
    m=k.ModelNew(B*T,K,N).float(); x=torch.randn(B,T,K)
    with torch.no_grad():
        e0=(m(x)-ref(x,m.weight)).abs().max().item(); ptr0=m.weight.data_ptr()
        m.weight.mul_(-3.0).add_(torch.randn_like(m.weight)*0.05)   # in-place update (e.g. optimizer step / load_state_dict copy_)
        same_ptr=m.weight.data_ptr()==ptr0
        y=m(x); r=ref(x,m.weight); e1=(y-r).abs().max().item()
        rel=((y-r).norm()/r.norm()).item(); ok=torch.allclose(y,r,rtol=1e-3,atol=1e-3)
        m.weight.copy_(torch.randn_like(m.weight)*0.02)   # load_state_dict-style
        e2=(m(x)-ref(x,m.weight)).abs().max().item()
    out['runs'].append(dict(seed=seed,max_abs_err_before=e0,data_ptr_unchanged=same_ptr,max_abs_err_after_inplace=e1,
        rel_err_after_inplace=rel,allclose_1e3_after=ok,max_abs_err_after_copy_=e2))
json.dump(out,open('fp8_stale_cache_results.json','w'),indent=1);print(json.dumps(out,indent=1))
