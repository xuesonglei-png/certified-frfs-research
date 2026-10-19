"""Report both rules and all available native control contrasts, without selection."""
import json
import numpy as np
from run_anchor_guarded_selector_v5 import FOLDER
from run_repair_relation_selector_v4 import ROOT,NAMES,OUT,digest
from anchor_guarded_selector_v5 import RULES

def main():
    rows=[];missing=[]
    for n in NAMES:
        p=FOLDER/(n+'.json')
        if not p.exists():missing.append(n);continue
        s=json.loads(p.read_text())
        if not s['complete']:missing.append(n);continue
        r={'dataset':n};a=FOLDER/(n+'-audit.json');audit=json.loads(a.read_text()) if a.exists() else {}
        r['audit_passed']=bool(audit.get('passed') and audit.get('artifact_sha256')==digest(p))
        for rule in RULES:
            for metric in ('clean_BA','query_certified_BA','query_coverage','macro_F1','balanced_logloss'):
                r[rule+'_'+metric]=float(np.mean([v['scores'][rule][metric] for v in s['rows']]))
            r[rule+'_removed']=float(np.mean([1-len(v['fits'][rule]['selected']['features'])/s['d'] for v in s['rows']]))
            r[rule+'_perturbed_BA']=float(np.mean([v['perturbed_BA'][rule] for v in s['rows']]))
        r['full_clean_BA']=float(np.mean([v['full']['clean_BA'] for v in s['rows']]))
        r['previous_v4_clean_BA']=float(np.mean([v['previous_v4']['clean_BA'] for v in s['rows']]))
        b=ROOT/'results/repair-relation-controls-v4'/(n+'.json');controls=json.loads(b.read_text()) if b.exists() else None
        r['native_baselines_complete']=bool(controls and controls['complete'])
        if r['native_baselines_complete']:
            for m in controls['rows'][0]['scores']:
                if '_matched_' in m:continue
                vals=[v['scores'][m] for v in controls['rows']]
                r[m+'_clean_BA']=float(np.mean([v['clean_BA'] for v in vals])) if all(v is not None for v in vals) else None
        rows.append(r)
    contrasts={}
    methods=sorted({k[:-9] for r in rows for k in r if k.endswith('_clean_BA')}-set(RULES))
    for rule in RULES:
        contrasts[rule]={}
        for m in methods:
            ds=[r[rule+'_clean_BA']-r[m+'_clean_BA'] for r in rows if r.get(m+'_clean_BA') is not None]
            contrasts[rule][m]={'datasets':len(ds),'mean_delta_pp':100*float(np.mean(ds)) if ds else None,
                'wins':sum(v>1e-12 for v in ds),'ties':sum(abs(v)<=1e-12 for v in ds),'losses':sum(v< -1e-12 for v in ds)}
    result={'primary':'paired_anchor','datasets':len(rows),'complete':not missing,'missing':missing,
        'rows':rows,'contrasts':contrasts,'mean_removed':{r:float(np.mean([v[r+'_removed'] for v in rows])) if rows else None for r in RULES},
        'goal_achieved':False,'classification_superiority_established':False,'scope':'descriptive_exposed_development'}
    (FOLDER/'development-summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))

if __name__=='__main__':main()
