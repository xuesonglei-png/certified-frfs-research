"""No held-out scores: replay training-only rejected v4 supports with more friends."""
import argparse,json,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.majority_metric import MajorityTerms
from expanded_majority_guard_v1 import expanded_bounds
from run_repair_relation_selector_v4 import ROOT,OUT,NAMES,digest

FOLDER=ROOT/'results/expanded-majority-admission-v1'

def run(name):
    threadpool_limits(1);path=OUT/(name+'.json');saved=json.loads(path.read_text());assert saved['complete']
    sha=digest(path);X,y=load_dataset(name);reports=[];start=time.perf_counter()
    for outer in saved['rows']:
        ri=np.array(outer['train_indices']);raw,labels=X[ri],y[ri]
        for fold,b in enumerate(outer['fit']['inner']):
            ii=np.array(b['fit_indices']);A,_=preprocess_fit(raw[ii],raw[ii]);truth=labels[ii];d=A.shape[1]
            uniform=np.ones(d)/np.sqrt(d)
            prot_uniform=MajorityTerms(A,truth,weights=uniform,eps=.01).bounds(range(d),allowed_enemies=2)>1e-10
            for family,part in b['families'].items():
                ref=np.array(part['reference_weights'])
                prot_family=MajorityTerms(A,truth,weights=ref,eps=.01).bounds(range(d),allowed_enemies=2)>1e-10
                candidates={tuple(c['features']):c for r in part['candidates'] for c in r['trace']
                    if not c['joint_feasible'] and all(c[k]['dependency_certificate']['upper_bound']<=.1 for k in ('certificate','family_certificate'))
                    and c['classwise']['feasible'] and c['family_classwise']['feasible']}
                for fs,c in candidates.items():
                    w=np.array(c['weights']);small=expanded_bounds(A[:,fs],truth,w,n_friends=3)
                    large=expanded_bounds(A[:,fs],truth,w,n_friends=15)
                    assert np.all(large>=small-1e-12)
                    accepted=[];retained=[]
                    for prot,k in ((prot_uniform,'certificate'),(prot_family,'family_certificate')):
                        groups=[(truth==label)&prot for label in np.unique(truth)]
                        counts=[int(((large>1e-10)&g).sum()) for g in groups];required=c[k]['required']
                        accepted.append(all(a>=r for a,r in zip(counts,required)));retained.append(counts)
                    reports.append({'outer_fold':outer['fold'],'inner_fold':fold,'family':family,'features':list(fs),
                        'newly_admitted':all(accepted),'retained':retained,
                        'additional_correct_training_certificates':int(np.sum((large>1e-10)&(small<=1e-10)))})
            print(name,outer['fold'],fold,'training-only evaluated',len(reports),'newly admitted',sum(r['newly_admitted'] for r in reports),flush=True)
    FOLDER.mkdir(exist_ok=True);assert digest(path)==sha
    result={'dataset':name,'rows':reports,'evaluated':len(reports),'newly_admitted':sum(r['newly_admitted'] for r in reports),
        'additional_certificates':sum(r['additional_correct_training_certificates'] for r in reports),
        'source_sha256':sha,'runner_sha256':digest(__file__),'guard_sha256':digest(ROOT/'scripts/expanded_majority_guard_v1.py'),
        'seconds':time.perf_counter()-start,'new_outer_scoring':False,'goal_achieved':False}
    (FOLDER/(name+'.json')).write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items() if k!='rows'},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--only',choices=NAMES);args=p.parse_args()
    for n in ((args.only,) if args.only else NAMES):run(n)
