import os,sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
"""strict_checker on the CPU-runnable 'no custom kernel' passes, against upstream KernelBench references
(refs/ fetched from ScalingIntelligence/KernelBench main). Small shapes for CPU; real ModelNew code."""
import importlib.util,io,contextlib,json,torch
from strict_checker import strict_check
S=os.environ.get('WAFER_DATA','wafer-data')+'/kernel-arena/exports/kernelbench-hip-mi300x/solutions'
def load(p):
    with contextlib.redirect_stdout(io.StringIO()):
        s=importlib.util.spec_from_file_location('m'+str(abs(hash(p))),p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    return m
cases=[('level2_98_Matmul_AvgPool_GELU_Scale_Max','grok_4.1_min',os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','refs/')+'98_Matmul_AvgPool_GELU_Scale_Max.py',(256,256,16,2.0),
        lambda sh,g:[torch.rand(sh,256,generator=g)],[8,1,33],{'linear.weight':'matmul.weight','linear.bias':'matmul.bias'}),
       ('level2_73_Conv2d_BatchNorm_Scaling','gemini_3_flash',os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','refs/')+'73_Conv2d_BatchNorm_Scaling.py',(3,8,3,2.0),
        lambda sh,g:[torch.rand(sh,3,16,16,generator=g)],[2,1,5],None),
       ('level2_40_Matmul_Scaling_ResidualAdd','grok_4.1',os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','refs/')+'40_Matmul_Scaling_ResidualAdd.py',(64,32,0.5),
        lambda sh,g:[torch.rand(sh,64,generator=g)],[8,1,33],{'linear.weight':'matmul.weight','linear.bias':'matmul.bias'}),
       ('level2_40_Matmul_Scaling_ResidualAdd','opus_4.5_min',os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','refs/')+'40_Matmul_Scaling_ResidualAdd.py',(64,32,0.5),
        lambda sh,g:[torch.rand(sh,64,generator=g)],[8,1,33],None)]
out={}
for task,model,ref,ia,mk,shapes,pm in cases:
    R=load(ref).Model; N=load(f'{S}/{task}/{model}/kernel.py').ModelNew
    for pw in (False,True):
        r=strict_check(R,N,ia,mk,shapes,perturb_weights=pw,param_map=pm,rtol=1e-4,atol=1e-4)
        out[f'{task}/{model}/perturb={pw}']=dict(passed=r.passed,n_checks=r.n_checks,n_failures=len(r.failures),first_failures=r.failures[:2])
json.dump(out,open('strict_check_cpu_candidates.json','w'),indent=1);print(json.dumps(out,indent=1))
