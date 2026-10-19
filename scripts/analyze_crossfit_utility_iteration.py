"""Frozen five-dataset pilot analysis; all specified clean comparisons together."""
import json
from run_decision_iteration import ROOT
from analyze_crossfit_majority_expansion import analyze


def main():
    directory=ROOT/'results/predictive-iteration36'
    files=sorted(directory.glob('*-s101-f*.json'));assert len(files)==15
    grouped={}
    for file in files:
        obj=json.loads(file.read_text())
        for r in obj['rows']:
            grouped.setdefault(r['method'],{}).setdefault(obj['dataset'],[]).append({**r['metrics'],'removed':1-len(r['features'])/obj['d']})
        old=json.loads((ROOT/'results/predictive-iteration35'/file.name).read_text())
        for method,alias in [('protected','loo_protected'),('utility','loo_utility')]:
            r=next(r for r in old['rows'] if r['method']==method)
            grouped.setdefault(alias,{}).setdefault(obj['dataset'],[]).append({**r['metrics'],'removed':1-len(r['features'])/obj['d']})
        for r in obj['rows']:
            if r['method'] in ('protected','utility'):continue
            prior=next(p for p in old['rows'] if p['method']==r['method'])
            for key in ('features','weights','chosen','metrics'):assert r[key]==prior[key]
    analysis=analyze(grouped,[]);means={m:r['means'] for m,r in analysis['methods'].items()}
    delta=means['protected']['clean']-means['loo_protected']['clean']
    result={'status':'repeated_development_not_confirmation','analysis':analysis,
            'primary_mean_improvement_over35':delta,'accuracy_gate_at_least_half_point':delta>=.005,
            'compression_no_collapse':means['protected']['removed']>=means['loo_protected']['removed']-.02}
    (directory/'crossfit-analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'means':means,'delta':delta,'accuracy_gate':result['accuracy_gate_at_least_half_point'],
                      'compression_gate':result['compression_no_collapse'],'pairs':analysis['pairs']},indent=2))

if __name__=='__main__':main()
