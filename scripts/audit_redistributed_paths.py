"""Recompute all admitted changed-metric nodes and sample shared perturbations."""
import argparse,json,hashlib
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.redistributed_loss import transfer_weights,redistributed_loss_bound
from certified_frfs.majority_metric import MajorityTerms
from certified_frfs.core import gamma


def audit(A,y,wf,path,seed):
    original=MajorityTerms(A,y,weights=wf).bounds(range(A.shape[1]),allowed_enemies=2)>1e-10
    groups=[original&(y==label) for label in np.unique(y)]
    required=[int(np.ceil(.9*g.sum()-1e-12)) for g in groups]
    assert required==path['required'];rng=np.random.default_rng(seed)
    tables=[np.clip(A+rng.uniform(-.01,.01,A.shape),0,1) for _ in range(3)]
    count=0
    for fs in path['candidates']:
        wr=transfer_weights(A,fs,wf);dep=redistributed_loss_bound(A,y,fs,wf,wr)
        assert dep['upper_bound']<=.1
        bound=MajorityTerms(A[:,fs],y,weights=wr).bounds(range(len(fs)),allowed_enemies=2)
        assert all(int(((bound>1e-10)&g).sum())>=r for g,r in zip(groups,required))
        wide=np.zeros(A.shape[1]);wide[fs]=wr
        for Z in tables:
            assert gamma(Z,y,weights=wf)-gamma(Z,y,fs,weights=wide)<=dep['upper_bound']+1e-10
            count+=1
    return len(path['candidates']),count


def main(iteration=38):
    threadpool_limits(1);directory=ROOT/f'results/predictive-iteration{iteration}';files=sorted(directory.glob('*-s101-f*.json'))
    assert len(files)==5,'pilot incomplete';rows=[]
    for file in files:
        o=json.loads(file.read_text());X,y=load_dataset(o['dataset']);tr=np.array(o['train_indices']);nodes=checks=paths=0
        for index,rec in enumerate(o['inner_splits']):
            ii=np.array(rec['train']);A,_=preprocess_fit(X[tr][ii],X[tr][ii])
            for fit in rec['redistributed_fits']:
                if fit['method']!='protected':continue
                a,b=audit(A,y[tr][ii],np.array(rec['weights'][fit['family']]),fit['path'],950+index)
                nodes+=a;checks+=b;paths+=1
        r=next(r for r in o['rows'] if r['method']=='protected');A,_=preprocess_fit(X[tr],X[tr])
        a,b=audit(A,y[tr],np.array(r['full_weights']),r['path'],954);nodes+=a;checks+=b;paths+=1
        rows.append({'file':file.name,'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'paths':paths,'nodes':nodes,'sampled_checks':checks})
        print('admitted nodes checked',file.name,flush=True)
    result={'scope':'all_admitted_training_nodes_not_search_regeneration_or_test_generalization',
        'rows':rows,'totals':{k:sum(r[k] for r in rows) for k in ['paths','nodes','sampled_checks']},'sampled_violations':0}
    (directory/'admitted-path-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result['totals']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--iteration',type=int,choices=[38,39],default=38);main(p.parse_args().iteration)
