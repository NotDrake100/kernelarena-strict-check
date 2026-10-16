"""Build charts from results/gpu/*.json (A100 run). Usage: python scripts/make_charts.py results/gpu charts"""
import json,glob,os,sys
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R,C=sys.argv[1],sys.argv[2]; os.makedirs(C,exist_ok=True)
ENV=json.load(open(f'{R}/env.json')); GPU=ENV['gpu'] or 'CPU'
recs=[json.load(open(f)) for f in sorted(glob.glob(f'{R}/level*.json'))]
recs=[r for r in recs if 'strict_fixed' in r]
lab=lambda r: f"{r['task'].split('_',1)[0]}_{r['task'].split('_')[1]} / {r['model']}"
# 1 pass/fail
fig,ax=plt.subplots(figsize=(9,0.42*len(recs)+1.6))
cols=['Leaderboard\n(original check)','Strict:\nfresh inputs, fixed weights','Strict:\n+ in-place weight updates']
for i,r in enumerate(recs):
    vals=[bool(r['claimed'] and r['claimed']['passed']),r['strict_fixed']['passed'],r['strict_inplace']['passed']]
    for j,v in enumerate(vals):
        ax.add_patch(plt.Rectangle((j,i),0.95,0.9,color='#3a9d5d' if v else '#d64545'))
        ax.text(j+0.47,i+0.45,'PASS' if v else 'FAIL',ha='center',va='center',color='white',fontsize=9,weight='bold')
ax.set_xlim(0,3); ax.set_ylim(len(recs),0); ax.set_xticks([0.47,1.47,2.47]); ax.set_xticklabels(cols,fontsize=9)
ax.set_yticks([i+0.45 for i in range(len(recs))]); ax.set_yticklabels([lab(r) for r in recs],fontsize=8)
ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
ax.set_title(f'KernelArena passes re-checked on {GPU} (rtol=atol=1e-3)',fontsize=11)
plt.tight_layout(); plt.savefig(f'{C}/pass_fail.png',dpi=150); plt.close()
# 2 stale error
st=[r for r in recs if r['strict_fixed']['passed'] and not r['strict_inplace']['passed']]
if st:
    fig,ax=plt.subplots(figsize=(8,3.6))
    import numpy as np
    before=[r['strict_fixed']['max_abs_err_max'] for r in st]; after=[r['strict_inplace']['max_abs_err_max'] for r in st]
    x=np.arange(len(st)); ax.bar(x-0.2,[max(b,1e-9) for b in before],0.4,label='weights unchanged (max over 27 checks)',color='#3a9d5d')
    ax.bar(x+0.2,after,0.4,label='after in-place weight update (max over 27 checks)',color='#d64545')
    for i,(b,a_) in enumerate(zip(before,after)):
        ax.text(i-0.2,max(b,1e-9)*1.6,'0' if b==0 else f'{b:.0e}',ha='center',fontsize=7)
        ax.text(i+0.2,a_*1.6,f'{a_:.3g}',ha='center',fontsize=7)
    ax.set_ylim(1e-10,1e4)
    ax.axhline(1e-3,ls='--',color='k',lw=1); ax.text(len(st)-0.5,1.3e-3,'tolerance 1e-3',ha='right',fontsize=8)
    ax.set_yscale('log'); ax.set_xticks(x); ax.set_xticklabels([lab(r) for r in st],fontsize=7,rotation=12)
    ax.set_ylabel('max |ModelNew - reference|'); ax.set_title(f'Stale weight caches: error after an in-place weight change ({GPU})',fontsize=10)
    ax.legend(fontsize=7); plt.tight_layout(); plt.savefig(f'{C}/stale_cache_error.png',dpi=150); plt.close()
# 3 timing
tm=[r for r in recs if r.get('timing_ms') and r['claimed'] and r['claimed'].get('speedup')]
fig,ax=plt.subplots(figsize=(9,0.42*len(tm)+1.5))
import numpy as np
y=np.arange(len(tm))
ax.barh(y-0.2,[r['claimed']['speedup'] for r in tm],0.4,label='leaderboard speedup (MI300X, as published)',color='#888')
ax.barh(y+0.2,[r['timing_ms']['measured_speedup'] for r in tm],0.4,label=f'measured on {GPU}: submission with HIP code replaced/absent vs reference',color='#2f6db5')
ax.axvline(1,color='k',lw=1); ax.set_yticks(y); ax.set_yticklabels([lab(r)+(' *' if 'stand-in' in r['port_note'] else '') for r in tm],fontsize=8); ax.invert_yaxis()
ax.set_xscale('log'); ax.set_xlabel('speedup vs reference (log scale)'); ax.legend(fontsize=7,loc='lower right')
ax.set_title('How much speedup comes from PyTorch built-ins / Python-level changes\n(* = HIP kernel replaced by a PyTorch stand-in)',fontsize=10)
plt.tight_layout(); plt.savefig(f'{C}/timing.png',dpi=150); plt.close()
print('charts ok',len(recs),len(st),len(tm))
