import os,sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
"""Apply strict_checker to the REAL opus_4.5 FP8 ModelNew on CPU (PyTorch fallback path for inputs;
original weight-cache code). RefFP8 = assumed reference: FP8 quant-dequant of x and CURRENT weight each call."""
import importlib.util,io,contextlib,json,torch,torch.nn as nn
from strict_checker import strict_check
P=os.environ.get('WAFER_DATA','wafer-data')+'/kernel-arena/exports/kernelbench-hip-mi300x/solutions/level4_104_FP8_Matmul/opus_4.5/kernel.py'
with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
    s=importlib.util.spec_from_file_location('k',P);k=importlib.util.module_from_spec(s);s.loader.exec_module(k)
def qd(t,f=448.0):
    sc=f/t.abs().max().clamp(min=1e-12); return ((t*sc).clamp(-f,f).to(torch.float8_e4m3fn).to(t.dtype))/sc
class RefFP8(nn.Module):
    def __init__(s,M,K,N): super().__init__(); s.K,s.N=K,N; s.weight=nn.Parameter(torch.randn(K,N)*0.02)
    def forward(s,x): B,T,_=x.shape; return (qd(x.reshape(-1,s.K))@qd(s.weight)).reshape(B,T,s.N)
mk=lambda sh,g:[torch.randn(*sh,generator=g)]
out={}
for name,pw in [('single_weights_only',False),('inplace_weight_updates',True)]:
    r=strict_check(RefFP8,k.ModelNew,(128,256,128),mk,[(2,64,256),(1,1,256),(3,17,256)],perturb_weights=pw)
    out[name]=dict(passed=r.passed,n_checks=r.n_checks,n_failures=len(r.failures),first_failures=r.failures[:3])
json.dump(out,open('strict_check_fp8_opus.json','w'),indent=1);print(json.dumps(out,indent=1))
