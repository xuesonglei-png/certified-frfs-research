import json,hashlib
import numpy as np
from threadpoolctl import threadpool_limits
from diagnose_bound_repair_pool import ROOT,FOLDER
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit,validation_scores
from certified_frfs.query_probability import QueryProbabilityObjective
from certified_frfs.feasible_transport import weight_certificate


def main():
    threadpool_limits(1);counts={'splits':0,'selected_certificates':0,'scores':0,'repaired_selected':0}
    repairs=[json.loads(p.read_text()) for p in (ROOT/'results/majority-bound-search').glob('*.json') if not p.name.startswith('complete')]
    for path in sorted(FOLDER.glob('*.json')):
        if path.name.startswith('complete'):continue
        p=json.loads(path.read_text());name=p['dataset'];prior_path=ROOT/'results/capped-sparse-extension'/path.name
        assert hashlib.sha256(prior_path.read_bytes()).hexdigest()==p['prior_sha256']
        prior=json.loads(prior_path.read_text());X,y=load_dataset(name);full=np.ones(p['d'])/np.sqrt(p['d'])
        for index,s in enumerate(p['splits']):
            old=prior['splits'][index];ri,vi=np.array(s['reference_indices']),np.array(s['validation_indices'])
            assert ri.tolist()==old['reference_indices'] and vi.tolist()==old['validation_indices'] and set(ri).isdisjoint(vi)
            A,B=preprocess_fit(X[ri],X[vi]);obj=QueryProbabilityObjective(X[ri],y[ri]);expected=[]
            for c in old['fit']['candidates']:
                if c['certificate']['feasible']:expected.append(('old',c['features'],c['weights']))
            for r in sorted(repairs,key=lambda r:(r['task']['group'],r['task']['dataset'],r['task']['split'],r['task']['candidate'])):
                t=r['task'];fit=r['fit']
                if t['group']=='original15' and t['dataset']==name and t['split']==index and fit['certificate']['feasible']:
                    v=np.zeros(p['d']);v[fit['features']]=fit['weights'];expected.append(('repair',fit['features'],v.tolist()))
            assert len(expected)==len(s['pool'])
            assert {(origin,tuple(fs),tuple(w)) for origin,fs,w in expected}=={(c['origin'],tuple(c['features']),tuple(c['weights'])) for c in s['pool']}
            for c in s['pool']:assert abs(obj(np.array(c['weights'])/full.sum())[0]-c['objective'])<1e-10
            best=min(s['pool'],key=lambda c:(c['objective'],len(c['features'])))
            chosen=min((c for c in s['pool'] if c['objective']<=best['objective']+.01),key=lambda c:(len(c['features']),c['objective']))
            assert chosen==s['selected'];w=np.array(chosen['weights']);fs=chosen['features']
            assert weight_certificate(A,y[ri],fs,full,w[fs])['feasible']
            metrics=validation_scores(A,y[ri],B,y[vi],fs,w,144+index,joint=True)
            assert all(abs(metrics[k]-s['metrics'][k])<1e-12 for k in metrics)
            counts['splits']+=1;counts['selected_certificates']+=1;counts['scores']+=2;counts['repaired_selected']+=chosen['origin']=='repair'
    (FOLDER/'complete-audit.json').write_text(json.dumps(counts,indent=2),encoding='utf-8');print(json.dumps(counts,indent=2),flush=True)


if __name__=='__main__':main()
