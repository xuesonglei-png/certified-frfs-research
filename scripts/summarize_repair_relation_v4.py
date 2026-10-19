"""Dataset-balanced report retaining missing baselines and failed results."""
import json,csv
from pathlib import Path
import numpy as np
from run_repair_relation_selector_v4 import ROOT,OUT,NAMES,digest

def main():
    rows=[];missing=[]
    for n in NAMES:
        p=OUT/(n+'.json')
        if not p.exists():missing.append(n);continue
        s=json.loads(p.read_text())
        if not s['complete']:missing.append(n);continue
        base=ROOT/'results/repair-relation-controls-v4'/(n+'.json')
        controls=json.loads(base.read_text()) if base.exists() else None
        audit=OUT/(n+'-audit.json');a=json.loads(audit.read_text()) if audit.exists() else {}
        r={'dataset':n,'removed':float(np.mean([1-len(x['fit']['selected']['features'])/s['d'] for x in s['rows']])),
           'fallbacks':sum(x['fit']['selected']['uniform_fallback'] for x in s['rows']),
           'full_supports':sum(len(x['fit']['selected']['features'])==s['d'] for x in s['rows']),
           'audit_passed':bool(a.get('passed') and a.get('artifact_sha256')==digest(p)),
           'baseline_complete':bool(controls and controls['complete'])}
        for m in ('proposed','full','matched_uniform'):
            for k in ('clean_BA','query_certified_BA','query_coverage','balanced_query_coverage','macro_F1','balanced_logloss'):
                r[m+'_'+k]=float(np.mean([x[m][k] for x in s['rows']]))
            r[m+'_perturbed_BA']=float(np.mean([x['perturbed_BA'][m] for x in s['rows']]))
        previous=ROOT/'results/selective-support-selector-v3'/(n+'.json')
        old=json.loads(previous.read_text())
        r['previous_v3_clean_BA']=float(np.mean([x['proposed']['clean_BA'] for x in old['rows']]))
        if controls and controls['complete']:
            for m in controls['rows'][0]['scores']:
                vals=[x['scores'][m] for x in controls['rows']]
                r[m+'_clean_BA']=float(np.mean([v['clean_BA'] for v in vals])) if all(v is not None for v in vals) else None
                r[m+'_query_certified_BA']=float(np.mean([v['query_certified_BA'] for v in vals])) if all(v is not None for v in vals) else None
        rows.append(r)
    contrasts={}
    if rows:
        methods=sorted({k[:-len('_clean_BA')] for r in rows for k in r if k.endswith('_clean_BA')}-{'proposed'})
        for m in methods:
            vals=[r['proposed_clean_BA']-r[m+'_clean_BA'] for r in rows if r.get(m+'_clean_BA') is not None]
            contrasts[m]={'available_datasets':len(vals),'mean_delta_pp':100*float(np.mean(vals)) if vals else None,
                'wins':sum(v>1e-12 for v in vals),'losses':sum(v< -1e-12 for v in vals),'ties':sum(abs(v)<=1e-12 for v in vals)}
    summary={'complete':len(rows)==len(NAMES),'missing':missing,'datasets':len(rows),'rows':rows,
        'contrasts':contrasts,'mean_removed':float(np.mean([r['removed'] for r in rows])) if rows else None,
        'all_method_audited':bool(rows) and all(r['audit_passed'] for r in rows) and not missing,
        'all_baselines_complete':bool(rows) and all(r['baseline_complete'] for r in rows) and not missing,
        'classification_superiority_established':False,'goal_achieved':False,
        'scope':'descriptive_exposed_development_not_confirmation_no_success_assumed'}
    (OUT/'development-summary.json').write_text(json.dumps(summary,indent=2))
    fields=sorted({k for r in rows for k in r})
    with (OUT/'dataset-means.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))

if __name__=='__main__':main()
