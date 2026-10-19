"""Quantify dual-constraint operating points without changing the method."""
import json
import numpy as np
from run_decision_iteration import ROOT
from analyze_crossfit_majority_expansion import analyze


def main():
    directory=ROOT/'results/predictive-iteration34';files=sorted(directory.glob('*-s101-f*.json'))
    assert len(files)==15;grouped={};diagnostics=[]
    for file in files:
        obj=json.loads(file.read_text());old=json.loads((ROOT/'results/predictive-iteration33'/file.name).read_text())
        for method,source in [('protected',obj),('majority_only',old)]:
            r=next(r for r in source['rows'] if r['method']=='protected')
            grouped.setdefault(method,{}).setdefault(obj['dataset'],[]).append({**r['metrics'],'removed':1-len(r['features'])/obj['d']})
        for r in obj['rows']:
            if r['method']=='protected':continue
            grouped.setdefault(r['method'],{}).setdefault(obj['dataset'],[]).append({**r['metrics'],'removed':1-len(r['features'])/obj['d']})
        r=next(r for r in obj['rows'] if r['method']=='protected');attempts=[a for step in r['path']['trace'] for a in step['certification_attempts']]
        diagnostics.append({'dataset':obj['dataset'],'fold':obj['fold'],
            'final_fuzzy_loss_bound':r['certificate']['dependency_certificate']['upper_bound'],
            'attempts':len(attempts),'majority_rejections':sum(not a['majority_feasible'] for a in attempts),
            'dependency_rejections_after_majority_pass':sum(a['majority_feasible'] and not a['feasible'] for a in attempts),
            'full_subset':len(r['features'])==obj['d'],'fit_seconds':r['fit_seconds']})
    assert all(len(rows)==3 for method in grouped.values() for rows in method.values())
    analysis=analyze(grouped,[])
    result={'status':'paired_development_dual_constraint_diagnostic_not_confirmation',
        'analysis':analysis,'diagnostics':diagnostics,
        'full_subset_models':sum(r['full_subset'] for r in diagnostics),
        'mean_loss_upper_bound':float(np.mean([r['final_fuzzy_loss_bound'] for r in diagnostics]))}
    (directory/'joint-cost-analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'means':{m:r['means'] for m,r in analysis['methods'].items()},
        'full_subset_models':result['full_subset_models'],'mean_loss_upper_bound':result['mean_loss_upper_bound'],
        'pairs':analysis['pairs']},indent=2))

if __name__=='__main__':main()
