"""Separate nominal constraint failures from certificate looseness, iteration22."""
import json
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.core import certificate


def main():
    threadpool_limits(1);rows=[]
    directory=ROOT/'results/predictive-iteration22'
    for path in sorted(directory.glob('*-s*-f*.json')):
        obj=json.loads(path.read_text());X,y=load_dataset(obj['dataset'])
        tr=np.array(obj['train_indices']);A,_=preprocess_fit(X[tr],X[tr])
        for row in obj['rows']:
            detail=row['detail'];w=np.array(detail['weights'])
            fs=detail['chosen']['features'];pre=certificate(A,y[tr],fs,weights=w,eps=.01)
            exact=None
            if pre['nominal_loss']<=.1 and pre['upper_bound']>.1:
                exact=certificate(A,y[tr],fs,weights=w,eps=.01,stability='exact')
            paths=detail.get('paths',[])
            trace=[c for p in paths for c in p['trace']]
            rejected=[c for c in trace if not c['feasible']]
            nominal_rejections=sum(c['loss_certificate']['nominal_loss']>.1+1e-10 for c in rejected)
            rows.append({'dataset':obj['dataset'],'fold':obj['fold'],'method':row['method'],
                'pre_restoration_certificate':pre,'exact_stability_certificate':exact,
                'restored':len(detail['restoration']),
                'rejected_discovery_deletions':len(rejected),
                'nominally_impossible_discovery_deletions':nominal_rejections})
    summary={}
    for method in sorted({r['method'] for r in rows}):
        group=[r for r in rows if r['method']==method]
        summary[method]={
            'models':len(group),
            'chosen_nominally_infeasible':sum(r['pre_restoration_certificate']['nominal_loss']>.1+1e-10 for r in group),
            'chosen_bound_infeasible':sum(r['pre_restoration_certificate']['upper_bound']>.1+1e-10 for r in group),
            'rescued_by_exact_stability':sum(r['exact_stability_certificate'] is not None and r['exact_stability_certificate']['upper_bound']<=.1+1e-10 for r in group),
            'restored_total':sum(r['restored'] for r in group),
            'rejected_discovery_deletions':sum(r['rejected_discovery_deletions'] for r in group),
            'nominally_impossible_discovery_deletions':sum(r['nominally_impossible_discovery_deletions'] for r in group)}
    (directory/'constraint-diagnostic.json').write_text(json.dumps({'summary':summary,'rows':rows,
        'status':'posthoc_mechanism_audit_not_new_predictive_method'},indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
