import json,hashlib
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit,validation_scores
from certified_frfs.query_probability import QueryProbabilityObjective
from certified_frfs.feasible_transport import weight_certificate

FOLDER=ROOT/'results/bound-repair-pool'


def main():
    threadpool_limits(1);FOLDER.mkdir(exist_ok=True)
    prior=json.loads((ROOT/'results/capped-sparse-extension/complete.json').read_text())
    long=json.loads((ROOT/'results/long-capped-sparse/complete.json').read_text())
    repairs=[json.loads(p.read_text()) for p in sorted((ROOT/'results/majority-bound-search').glob('*.json')) if not p.name.startswith('complete')]
    summary=[];chosen_count=0
    for base in prior['rows']:
        name=base['dataset'];out=FOLDER/f'{name}.json'
        if out.exists():raise FileExistsError(out)
        path=ROOT/'results/capped-sparse-extension'/f'{name}.json';p=json.loads(path.read_text());X,y=load_dataset(name);rows=[]
        for index,s in enumerate(p['splits']):
            ri,vi=np.array(s['reference_indices']),np.array(s['validation_indices']);A,B=preprocess_fit(X[ri],X[vi])
            obj=QueryProbabilityObjective(X[ri],y[ri]);pool=[];mass=np.sqrt(p['d']);full=np.ones(p['d'])/mass
            for c in s['fit']['candidates']:
                if c['certificate']['feasible']:pool.append({'origin':'old','features':c['features'],'weights':c['weights'],'objective':obj(np.array(c['weights'])/mass)[0]})
            for repair in repairs:
                t=repair['task'];fit=repair['fit']
                if t['group']=='original15' and t['dataset']==name and t['split']==index and fit['certificate']['feasible']:
                    v=np.zeros(p['d']);v[fit['features']]=fit['weights'];cert=weight_certificate(A,y[ri],fit['features'],full,np.array(fit['weights']))
                    assert cert==fit['certificate'] and cert['feasible']
                    pool.append({'origin':'repair','features':fit['features'],'weights':v.tolist(),'objective':obj(v/mass)[0],'certificate':cert})
            best=min(pool,key=lambda c:(c['objective'],len(c['features'])))
            c=min((c for c in pool if c['objective']<=best['objective']+.01),key=lambda c:(len(c['features']),c['objective']))
            metrics=validation_scores(A,y[ri],B,y[vi],c['features'],np.array(c['weights']),144+index,joint=True)
            chosen_count+=c['origin']=='repair'
            rows.append({'reference_indices':ri.tolist(),'validation_indices':vi.tolist(),'pool':pool,'selected':c,'metrics':metrics})
        out.write_text(json.dumps({'dataset':name,'d':p['d'],'splits':rows,'prior_sha256':hashlib.sha256(path.read_bytes()).hexdigest()},indent=2),encoding='utf-8')
        old_long=next(r for r in long['rows'] if r['dataset']==name)
        summary.append({'dataset':name,'utility':float(np.mean([(s['metrics']['clean']+s['metrics']['noise01'])/2 for s in rows])),
            'removed':float(np.mean([1-len(s['selected']['features'])/p['d'] for s in rows])),
            'control':{**base['utility'],'long':old_long['utility']}})
        print(name,'pool evaluation complete',flush=True)
    result={'rows':summary,'selected_repaired_splits':chosen_count,'mean_utility':float(np.mean([r['utility'] for r in summary])),
        'mean_removed':float(np.mean([r['removed'] for r in summary])),
        'comparisons':{m:{'gain':float(np.mean([r['utility']-r['control'][m] for r in summary])),
            'wins':sum(r['utility']>r['control'][m]+1e-12 for r in summary)} for m in ('compact','long','fgs2023','sfrip2025')},
        'scope':'all15_all44_fixed_training_pool_augmentation_diagnostic_not_complete_new_algorithm_or_confirmation'}
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)


if __name__=='__main__':main()
