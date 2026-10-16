"""GPU (or CPU smoke) run: strict checker + timing for KernelArena passes runnable on NVIDIA.
Usage: python gpu_run.py --sol SOLUTIONS_DIR --out OUTDIR [--small] [--only substr]
HIP kernels are NOT compiled: load_inline is replaced by a stub that returns PyTorch stand-ins for the
specific HIP functions listed in STANDINS (formulas copied from the HIP source); any other
load_inline call raises, which makes files with a fallback path use their own PyTorch fallback."""
import argparse,importlib.util,io,contextlib,json,os,time,statistics,traceback,types
import torch,torch.nn as nn
import torch.utils.cpp_extension as cppext
from strict_checker import strict_check
HERE=os.path.dirname(os.path.abspath(__file__))
def gelu_erf(x): return 0.5*x*(1+torch.erf(x*0.7071067811865476))
def gelu_tanh(x): return 0.5*x*(1+torch.tanh(0.7978845608028654*(x+0.044715*x**3)))
STANDINS={'gelu_scale_max_hip':lambda y,s:(gelu_erf(y)*s).max(dim=1).values,
          'gelu_scale_max':lambda y,s:(gelu_tanh(y)*s).max(dim=1).values,
          'post_matmul_fused_hip':lambda y,s:(gelu_erf(y)*s).max(dim=1).values,
          'new_gelu_forward':gelu_tanh}
def fake_load_inline(name=None,*a,functions=None,cuda_sources='',cpp_sources='',**k):
    j=lambda v: v if isinstance(v,str) else ''.join(v or [])
    src=j(cuda_sources)+j(cpp_sources)
    fns=functions if functions else [f for f in STANDINS if f+'(' in src]
    if isinstance(fns,str): fns=[fns]
    if fns and all(f in STANDINS for f in fns): return types.SimpleNamespace(**{f:STANDINS[f] for f in fns})
    raise RuntimeError(f'HIP extension {name} not compiled on this platform (functions={fns})')
def load(path):
    if not hasattr(torch.backends,'miopen'):   # compatibility shim: CUDA builds of PyTorch have no torch.backends.miopen
        torch.backends.miopen=types.SimpleNamespace(benchmark=False)
    env=os.environ.copy(); orig=cppext.load_inline; cppext.load_inline=fake_load_inline; buf=io.StringIO()
    try:
        with contextlib.redirect_stdout(buf),contextlib.redirect_stderr(buf):
            sp=importlib.util.spec_from_file_location('m%d'%abs(hash(path)),path); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    finally:
        cppext.load_inline=orig; os.environ.clear(); os.environ.update(env)
    return m,buf.getvalue()[-400:]
def qd(t,f=448.0):
    sc=f/t.abs().max().clamp(min=1e-12); return ((t*sc).clamp(-f,f).to(torch.float8_e4m3fn).to(t.dtype))/sc
class RefFP8(nn.Module):  # ASSUMED reference (Wafer's level4_104 reference is not public)
    def __init__(s,M,K,N,use_e4m3=True): super().__init__(); s.K,s.N=K,N; s.weight=nn.Parameter(torch.randn(K,N)*0.02)
    def forward(s,x): B,T,_=x.shape; return (qd(x.reshape(-1,s.K))@qd(s.weight)).reshape(B,T,s.N)
LIN={'linear.weight':'matmul.weight','linear.bias':'matmul.bias'}
P98=((8192,8192,16,2.0),[(1024,8192)],(256,256,16,2.0),[(8,256)],[1,17])
P40=((4096,4096,0.5),[(16384,4096)],(64,32,0.5),[(8,64)],[1,17])
P43=((768,8,0.0,0.0,1024),[(128,512,768)],(64,4,0.0,0.0,64),[(2,16,64)],[1,3])
P104=((16384,4096,4096),[(8,2048,4096)],(128,256,128),[(2,64,256)],[1,3])
CASES=[
 ('level2_73_Conv2d_BatchNorm_Scaling','gemini_3_flash','73_Conv2d_BatchNorm_Scaling.py',((8,64,3,2.0),[(128,8,128,128)],(3,8,3,2.0),[(2,3,16,16)],[1,17]),None,'pure PyTorch'),
 ('level2_98_Matmul_AvgPool_GELU_Scale_Max','grok_4.1_min','98_Matmul_AvgPool_GELU_Scale_Max.py',P98,LIN,'pure PyTorch'),
 ('level2_98_Matmul_AvgPool_GELU_Scale_Max','gpt_5.2','98_Matmul_AvgPool_GELU_Scale_Max.py',P98,None,'HIP gelu_scale_max_hip replaced by PyTorch stand-in (erf GELU*scale, max over dim 1); Python weight-cache logic unchanged'),
 ('level2_98_Matmul_AvgPool_GELU_Scale_Max','gpt_5.2_min','98_Matmul_AvgPool_GELU_Scale_Max.py',P98,None,'HIP gelu_scale_max replaced by PyTorch stand-in (tanh GELU as in the HIP source); Python weight-cache logic unchanged'),
 ('level2_98_Matmul_AvgPool_GELU_Scale_Max','gemini_3_flash','98_Matmul_AvgPool_GELU_Scale_Max.py',P98,None,'HIP post_matmul_fused_hip replaced by PyTorch stand-in (erf GELU*scale, max over dim 1); Python weight-cache logic unchanged'),
 ('level2_40_Matmul_Scaling_ResidualAdd','grok_4.1','40_Matmul_Scaling_ResidualAdd.py',P40,LIN,'pure PyTorch'),
 ('level2_40_Matmul_Scaling_ResidualAdd','opus_4.5_min','40_Matmul_Scaling_ResidualAdd.py',P40,None,'pure PyTorch'),
 ('level2_55_Matmul_MaxPool_Sum_Scale','grok_4.1_min','55_Matmul_MaxPool_Sum_Scale.py',((32768,32768,2,0.5),[(128,32768)],(64,64,2,0.5),[(8,64)],[1,17]),LIN,'pure PyTorch'),
 ('level1_9_Tall_skinny_matrix_multiplication_','gpt_5.2_min','9_Tall_skinny_matrix_multiplication_.py',None,None,'pure PyTorch'),
 ('level3_43_MinGPTCausalAttention','grok_4.1_min','43_MinGPTCausalAttention.py',P43,None,'pure PyTorch (SDPA)'),
 ('level3_43_MinGPTCausalAttention','kimi_k2','43_MinGPTCausalAttention.py',P43,None,'pure PyTorch'),
 ('level3_43_MinGPTCausalAttention','opus_4.5','43_MinGPTCausalAttention.py',P43,None,'pure PyTorch (SDPA)'),
 ('level3_43_MinGPTCausalAttention','opus_4.5_min','43_MinGPTCausalAttention.py',P43,None,'pure PyTorch (SDPA)'),
 ('level3_44_MiniGPTBlock','gpt_5.2','44_MiniGPTBlock.py',P43,None,'HIP new_gelu_forward replaced by PyTorch NewGELU stand-in (same tanh formula); attention is SDPA as submitted'),
 ('level4_104_FP8_Matmul','opus_4.5','ASSUMED_RefFP8',P104,None,"HIP input-quant extension not built, so the file's own PyTorch fallback runs; weight-cache code unchanged; reference ASSUMED"),
 ('level4_104_FP8_Matmul','gemini_3_flash','ASSUMED_RefFP8',P104,None,'torch.compile as submitted; reference ASSUMED'),
]
def bench(fn,inp,dev,warm=3,iters=10):
    with torch.no_grad():
        for _ in range(warm): fn(*inp)
        if dev=='cuda': torch.cuda.synchronize()
        ts=[]
        for _ in range(iters):
            if dev=='cuda':
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True); a.record(); fn(*inp); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
            else:
                t=time.perf_counter(); fn(*inp); ts.append((time.perf_counter()-t)*1e3)
    return statistics.median(ts)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--sol',required=True); ap.add_argument('--out',required=True); ap.add_argument('--small',action='store_true'); ap.add_argument('--only',default='')
    a=ap.parse_args(); os.makedirs(a.out,exist_ok=True); dev='cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    claimed={(r['task'],r['model']):r for r in json.load(open(os.path.join(os.path.dirname(a.sol),'index.json')))['results']}
    json.dump({'device':dev,'gpu':torch.cuda.get_device_name(0) if dev=='cuda' else None,'torch':torch.__version__,'cuda':torch.version.cuda,'small':a.small,'tf32':False},open(f'{a.out}/env.json','w'),indent=1)
    for (task,model,ref,P,pm,note) in CASES:
        key=f'{task}__{model}'
        if a.only and a.only not in key: continue
        if task.startswith('level4_104') and model=='gemini_3_flash':
            importlib.import_module('torch._dynamo').config.suppress_errors=True   # A100 (sm80) has no FP8 Triton codegen -> eager fallback
            note=note+'; A100 has no FP8 Triton support, so torch.compile falls back to eager (suppress_errors=True)'
        rec={'task':task,'model':model,'reference':ref,'port_note':note,'claimed':claimed.get((task,model))}; t0=time.time()
        try:
            mod,log=load(f'{a.sol}/{task}/{model}/kernel.py'); rec['load_log_tail']=log; New=mod.ModelNew
            if ref.startswith('ASSUMED'): Ref=RefFP8; refmod=None
            else: refmod,_=load(f'{HERE}/refs/{ref}'); Ref=refmod.Model
            if P is None:
                IA=tuple(refmod.get_init_inputs()); tmpl=tuple(tuple(t.shape) for t in refmod.get_inputs())
                if a.small: tmpl=((64,16),(16,64))
                shapes=[tmpl]; mk=lambda sh,g:[torch.rand(s,generator=g,device=g.device) for s in sh]
            else:
                ia,inp,ias,inps,bv=P; IA=ias if a.small else ia; base=(inps if a.small else inp)[0]
                rand=torch.randn if ref.startswith('ASSUMED') else torch.rand
                shapes=[base]+[(b,)+tuple(base[1:]) for b in bv]
                mk=lambda sh,g,rand=rand:[rand(sh,generator=g,device=g.device)]
            rec['init_args']=list(IA); rec['shapes']=[list(s) for s in shapes]
            for pw in (False,True):
                r=strict_check(Ref,New,IA,mk,shapes,seeds=(0,1,2),calls_per_seed=3,perturb_weights=pw,device=dev,param_map=pm,rtol=1e-3,atol=1e-3)
                errs=[e for e in r.max_abs_errs if e is not None]
                rec['strict_'+('inplace' if pw else 'fixed')]=dict(passed=r.passed,n_checks=r.n_checks,n_failures=len(r.failures),
                    max_abs_err_max=max(errs) if errs else None,max_abs_err_per_check=r.max_abs_errs,first_failures=r.failures[:3],
                    n_param_pairs=r.n_param_pairs,n_new_params=r.n_new_params)
                if dev=='cuda': torch.cuda.empty_cache()
            torch.manual_seed(0); R=Ref(*IA).to(dev).eval(); torch.manual_seed(0); N=New(*IA).to(dev).eval()
            g=torch.Generator(device=dev).manual_seed(0); x=mk(shapes[0],g)
            tr=bench(R,x,dev); tn=bench(N,x,dev)
            rec['timing_ms']=dict(reference=tr,modelnew=tn,measured_speedup=tr/tn); del R,N,x
        except Exception as e:
            rec['error']=f'{type(e).__name__}: {e}'; rec['traceback']=traceback.format_exc()[-1500:]
        rec['seconds']=time.time()-t0
        json.dump(rec,open(f'{a.out}/{key}.json','w'),indent=1)
        print(key,'| fixed:',rec.get('strict_fixed',{}).get('passed'),'| inplace:',rec.get('strict_inplace',{}).get('passed'),'|',rec.get('timing_ms'),'|',rec.get('error'),f"| {rec['seconds']:.0f}s",flush=True)
        if dev=='cuda': torch.cuda.empty_cache()
    if not a.only or 'fp8repro' in a.only or a.only=='all':
        M,K,N=(128,256,128) if a.small else (16384,4096,4096); out=[]
        for seed in range(3):
            torch.manual_seed(seed); x=torch.randn(M,K,device=dev).half(); w=(torch.randn(K,N,device=dev)*0.02).half()
            sc=448.0/x.abs().max().float(); hip=((x.float()*sc).clamp(-448,448)*(1/sc)).half()
            wq=qd(w).float(); r=qd(x).float()@wq; y=hip.float()@wq
            out.append(dict(seed=seed,shape=[M,K,N],max_abs_err=float((y-r).abs().max()),rel_fro_err=float((y-r).norm()/r.norm()),
              allclose_1e3=bool(torch.allclose(y,r,rtol=1e-3,atol=1e-3)),allclose_1e2=bool(torch.allclose(y,r,rtol=1e-2,atol=1e-2))))
        json.dump(dict(note='REPRODUCTION of the opus_4.5 level4_104 HIP fast_fp8_quant_dequant math (no FP8 cast) vs FP8 quant-dequant; not the HIP kernel itself',runs=out),open(f'{a.out}/fp8_hip_input_path_repro.json','w'),indent=1)
        print('fp8 repro',out,flush=True)
    print('ALL_DONE',flush=True)
if __name__=='__main__': main()
