import torch,torch.nn as nn
from strict_checker import strict_check,static_audit
class Ref(nn.Module):
    def __init__(s,K,N): super().__init__(); s.weight=nn.Parameter(torch.randn(K,N)*0.02)
    def forward(s,x): return x@s.weight
class Honest(Ref): pass
class StaleWeightCache(Ref):
    def forward(s,x):
        if getattr(s,'_p',None)!=s.weight.data_ptr(): s._c=s.weight.detach().clone(); s._p=s.weight.data_ptr()
        return x@s._c
class OutputCache(Ref):
    def forward(s,x):
        if not hasattr(s,'_o'): s._o=x@s.weight
        return s._o
class ShapeSpecialized(Ref):
    def forward(s,x): return x@s.weight if x.shape[0]==64 else torch.zeros(x.shape[0],s.weight.shape[1])
class InputMutator(Ref):
    def forward(s,x): y=x@s.weight; x.zero_(); return y
K,N=32,16
mk=lambda shape,g: [torch.randn(shape,K,generator=g)]
SH=[64,1,3,257]
def run(cls,shapes=SH,**kw): return strict_check(Ref,cls,(K,N),mk,shapes,**kw)
def test_honest_passes(): r=run(Honest); assert r.passed and r.n_checks==3*4*3, r.failures
def test_stale_weight_cache_caught(): assert not run(StaleWeightCache).passed
def test_stale_cache_missed_without_perturbation(): assert run(StaleWeightCache,perturb_weights=False).passed
def test_output_cache_caught(): assert not run(OutputCache).passed
def test_output_cache_missed_by_single_call(): assert strict_check(Ref,OutputCache,(K,N),mk,[64],seeds=(0,),calls_per_seed=1).passed
def test_shape_specialized_caught(): assert not run(ShapeSpecialized).passed
def test_shape_specialized_missed_on_nominal_shape(): assert run(ShapeSpecialized,shapes=[64]).passed
def test_input_mutation_caught(): r=run(InputMutator); assert any('mutated' in f for f in r.failures)
def test_audit_no_kernel():
    a=static_audit("import torch\nimport torch.nn.functional as F\nclass ModelNew:\n    def forward(self,x):\n        return F.gelu(torch.matmul(x,x))\n")
    assert a['verdict']=='no_custom_kernel' and a['forward_torch_ops']==['gelu','matmul']
def test_audit_kernel_used():
    a=static_audit("src='__global__ void k(){}'\nm=load_inline(name='x')\nclass ModelNew:\n    def forward(self,x):\n        return fused_ext.run(x)\n")
    assert a['verdict']=='custom_kernel_used'
def test_audit_kernel_unused():
    a=static_audit("src='__global__ void k(){}'\nclass ModelNew:\n    def forward(self,x):\n        return x.relu()\n")
    assert a['verdict']=='custom_kernel_unused_in_forward'
