"""Inspect every accepted dual-certified training path, sample both claims."""
import argparse,hashlib,json
import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.core import certificate,gamma
from certified_frfs.majority_metric import MajorityTerms
from certified_frfs.adaptive_witness import adaptive_bounds
from certified_frfs.shared_loss import shared_loss_bound


def audit(A,y,w,path,eta,seed,shared_loss=False):
    fixed=MajorityTerms(A,y,weights=w);protected=fixed.bounds(range(A.shape[1]),allowed_enemies=2)>1e-10
    groups=[(y==label)&protected for label in np.unique(y)]
    required=[int(np.ceil(.9*g.sum()-1e-12)) for g in groups];assert required==path['required']
    rng=np.random.default_rng(seed);tables=[np.clip(A+rng.uniform(-.01,.01,A.shape),0,1) for _ in range(5)]
    full_gamma=[gamma(Z,y,weights=w) for Z in tables];draws=objects=0
    for fs in path['candidates']:
        dep=certificate(A,y,fs,weights=w,eps=.01,method='hybrid')
        if shared_loss and dep['upper_bound']>eta:dep=shared_loss_bound(A,y,fs,w)
        assert dep['upper_bound']<=eta
        bounds=np.maximum(fixed.bounds(fs,allowed_enemies=2),adaptive_bounds(A,y,fs,w));mask=bounds>1e-10
        assert all(int((mask&g).sum())>=r for g,r in zip(groups,required))
        for Z,full in zip(tables,full_gamma):
            assert full-gamma(Z,y,fs,weights=w)<=dep['upper_bound']+1e-10
            D=cdist(Z[:,fs]*w[fs],Z[:,fs]*w[fs],'cityblock');np.fill_diagonal(D,np.inf)
            nn=np.argsort(D,axis=1,kind='stable')[:,:5]
            assert np.all(np.sum(y[nn]==y[:,None],axis=1)[mask]>=3)
            draws+=1;objects+=int(mask.sum())
    return len(path['candidates']),draws,objects


def main(iteration=34,resume=False):
    threadpool_limits(1);directory=ROOT/f'results/predictive-iteration{iteration}';rows=[]
    previous={}
    if resume and (directory/'joint-path-audit.json').exists():
        saved=json.loads((directory/'joint-path-audit.json').read_text())
        previous={r['file']:r for r in saved['rows']}
    for file in sorted(directory.glob('*-s101-f*.json')):
        cached=previous.get(file.name)
        if cached and cached['input_sha256']==hashlib.sha256(file.read_bytes()).hexdigest():
            rows.append(cached);continue
        obj=json.loads(file.read_text());X,y=load_dataset(obj['dataset']);tr=np.array(obj['train_indices']);eta=obj['dependency_eta']
        subsets=draws=objects=paths=0
        for index,record in enumerate(obj['inner_splits']):
            ii=np.array(record['train']);A,_=preprocess_fit(X[tr][ii],X[tr][ii])
            for fit in record['utility_fits']:
                if fit['method']!='protected':continue
                w=np.array(record['weights'][fit['family']]);a,b,c=audit(A,y[tr][ii],w,fit['path'],eta,204+index,shared_loss=obj.get('shared_loss',False))
                subsets+=a;draws+=b;objects+=c;paths+=1
        r=next(r for r in obj['rows'] if r['method']=='protected');A,_=preprocess_fit(X[tr],X[tr])
        a,b,c=audit(A,y[tr],np.array(r['weights']),r['path'],eta,305,shared_loss=obj.get('shared_loss',False))
        rows.append({'file':file.name,'input_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
            'verified_paths':paths+1,'verified_subsets':subsets+a,'sampled_dependency_checks':draws+b,
            'sampled_certified_object_checks':objects+c,'violations':0})
        print('joint paths checked',file.name,flush=True)
    result={'scope':'accepted_inner_and_final_paths_sampled_checks_not_exhaustive_or_generalization','rows':rows,
        'totals':{key:sum(r[key] for r in rows) for key in ('verified_paths','verified_subsets','sampled_dependency_checks','sampled_certified_object_checks','violations')}}
    (directory/'joint-path-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result['totals']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--iteration',type=int,choices=[34,35,36,37],default=34)
    p.add_argument('--resume',action='store_true');a=p.parse_args();main(a.iteration,a.resume)
