import os,sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import json,os,csv,collections
from strict_checker import static_audit
R=os.environ.get('WAFER_DATA','wafer-data')+'/kernel-arena/exports/kernelbench-hip-mi300x'; idx=json.load(open(f'{R}/index.json'))
rows=[]
for r in idx['results']:
    p=f"{R}/solutions/{r['task']}/{r['model']}/kernel.py"
    if not os.path.exists(p): continue
    a=static_audit(open(p).read()); rows.append({**r,'verdict':a['verdict'],'forward_torch_ops':';'.join(a['forward_torch_ops']),'fallback_branch':a['fallback_branch']})
csv.DictWriter(open('static_audit_all.csv','w',newline=''),fieldnames=rows[0].keys()).writeheader() if False else None
with open('static_audit_all.csv','w',newline='') as f: w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
P=[x for x in rows if x['passed']]
print(json.dumps({'files':len(rows),'passed':len(P),'verdicts_passed':collections.Counter(x['verdict'] for x in P),
 'verdicts_passed_speedup_gt1':collections.Counter(x['verdict'] for x in P if (x['speedup'] or 0)>1),
 'passed_null_speedup':sum(x['speedup'] is None for x in P)},indent=1))
