"""REPRODUCTION (not the HIP kernel itself): the HIP fast_fp8_quant_dequant_hip in opus_4.5/kernel.py
(lines ~120-130,167) computes clamp(x*scale,-max,max)*inv_scale in fp16 with NO cast to an FP8 type,
with scale=fp8_max/amax, so clamp never triggers and x is returned ~unchanged (no FP8 rounding).
This script mirrors that math in PyTorch and compares to FP8 quant-dequant reference."""
import torch,json
def qd(t,fmax=448.0,dt=torch.float8_e4m3fn):
    s=fmax/t.abs().max().clamp(min=1e-12); return ((t*s).clamp(-fmax,fmax).to(dt).to(t.dtype))/s
def hip_mirror(t,fmax=448.0):
    s=fmax/t.abs().max().float(); return ((t.float()*s).clamp(-fmax,fmax)*(1/s)).half()
res=[]
for seed in range(3):
  for (M,K,N) in [(128,256,128),(1024,4096,4096)]:
    torch.manual_seed(seed); x=torch.randn(M,K).half(); w=(torch.randn(K,N)*0.02).half()
    wq=qd(w).float(); r=qd(x).float()@wq; y=hip_mirror(x).float()@wq
    res.append(dict(seed=seed,shape=[M,K,N],input_identity_maxdiff=(hip_mirror(x).float()-x.float()).abs().max().item(),
      max_abs_err=(y-r).abs().max().item(),allclose_1e3=bool(torch.allclose(y,r,rtol=1e-3,atol=1e-3)),
      allclose_1e2=bool(torch.allclose(y,r,rtol=1e-2,atol=1e-2))))
json.dump(res,open('fp8_hip_path_repro_results.json','w'),indent=1);print(json.dumps(res,indent=1))
