"""Paired frozen five-task study: bound refinement versus old bound."""
import json
from run_decision_iteration import ROOT
from analyze_crossfit_majority_expansion import analyze


def main():
    directory=ROOT/'results/predictive-iteration35';files=sorted(directory.glob('*-s101-f*.json'));assert len(files)==15
    grouped={};changes=[]
    for file in files:
        obj=json.loads(file.read_text());old=json.loads((ROOT/'results/predictive-iteration34'/file.name).read_text())
        previous=json.loads((ROOT/'results/predictive-iteration33'/file.name).read_text())
        for method,source in [('protected',obj),('old_loss_bound',old),('majority_only',previous)]:
            r=next(r for r in source['rows'] if r['method']=='protected')
            grouped.setdefault(method,{}).setdefault(obj['dataset'],[]).append({**r['metrics'],'removed':1-len(r['features'])/obj['d']})
        for r in obj['rows']:
            if r['method']=='protected':continue
            prior=next(p for p in old['rows'] if p['method']==r['method'])
            for key in ('features','weights','chosen','metrics'):assert r[key]==prior[key]
            grouped.setdefault(r['method'],{}).setdefault(obj['dataset'],[]).append({**r['metrics'],'removed':1-len(r['features'])/obj['d']})
        new=next(r for r in obj['rows'] if r['method']=='protected');prior=next(r for r in old['rows'] if r['method']=='protected')
        attempts=[a for s in new['path']['trace'] for a in s['certification_attempts']]
        changes.append({'dataset':obj['dataset'],'fold':obj['fold'],'old_features':prior['features'],'new_features':new['features'],
            'clean_delta':new['metrics']['clean']-prior['metrics']['clean'],
            'refined_acceptances':sum(a['feasible'] and a['dependency_certificate'] is not None and 'shared_anchor_bound' in a['dependency_certificate'] for a in attempts),
            'new_loss_bound':new['certificate']['dependency_certificate']['upper_bound']})
    result={'status':'frozen_bound_refinement_repeated_development_no_confirmation',
        'analysis':analyze(grouped,[]),'changes':changes}
    (directory/'paired-bound-analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'means':{m:r['means'] for m,r in result['analysis']['methods'].items()},'changes':changes,
        'pairs':result['analysis']['pairs']},indent=2))

if __name__=='__main__':main()
