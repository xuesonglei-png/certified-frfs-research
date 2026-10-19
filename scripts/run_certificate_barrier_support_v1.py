"""Prospectively frozen, first-inner-training-only six-task mechanism screen."""
import hashlib,json,time
from collections import Counter
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.relation_support_selector_v1 import _candidate
from certified_frfs.majority_metric import MajorityTerms
from certificate_barrier_support_v1 import NominalProxy,search
from audit_selective_support_selector_v3 import check_candidate
from run_repair_relation_selector_v4 import ROOT,OUT,NAMES

FOLDER=ROOT/'results/certificate-barrier-support-v1'


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def freeze():
    files=list((ROOT/'src/certified_frfs').glob('*.py'))
    files += [ROOT/'scripts'/s for s in ('certificate_barrier_support_v1.py',
        'run_certificate_barrier_support_v1.py','audit_selective_support_selector_v3.py',
        'audit_relation_support_selector_v1.py','run_repair_relation_selector_v4.py')]
    files += [ROOT/'tests/test_certificate_barrier_support_v1.py',
              ROOT/'research/certificate-barrier-support-v1-protocol.md']
    for name in NAMES:
        files += [OUT/(name+'.json'),ROOT/'data/processed'/(name+'.npz')]
    hashes={p.relative_to(ROOT).as_posix():digest(p) for p in sorted(files)}
    FOLDER.mkdir(exist_ok=True);path=FOLDER/'source-freeze.json'
    if path.exists():assert json.loads(path.read_text())['hashes']==hashes
    else:path.write_text(json.dumps({'hashes':hashes,'time_unix':time.time(),
                     'new_outer_scoring':False},indent=2))
    return hashes


def independent_loo(A,y,features,weights):
    classes=np.unique(y);pred=[]
    for i in range(len(y)):
        D=np.array([sum(float(v)*abs(float(A[i,a])-float(A[j,a]))
                        for a,v in zip(features,weights,strict=True)) for j in range(len(y))])
        D[i]=np.inf;nn=np.argsort(D,kind='stable')[:min(5,len(y)-1)]
        counts=[sum(y[nn]==c) for c in classes];pred.append(classes[np.argmax(counts)])
    correct=np.asarray(pred)==y
    return float(np.mean([correct[y==c].mean() for c in classes]))


def run(name):
    threadpool_limits(1);hashes=freeze();path=FOLDER/(name+'.json')
    if path.exists():
        saved=json.loads(path.read_text());assert saved['source_hashes']==hashes
        if saved['complete']:return saved
    started=time.perf_counter();parent=json.loads((OUT/(name+'.json')).read_text())
    row=parent['rows'][0];block=row['fit']['inner'][0]
    X,y=load_dataset(name);outer=np.asarray(row['train_indices']);fit=np.asarray(block['fit_indices'])
    # These are enclosing and inner TRAIN indices only; no other score is read.
    A,_=preprocess_fit(X[outer[fit]],X[outer[fit]]);labels=y[outer[fit]];d=A.shape[1]
    part=block['families']['uniform_sqrt'];ref=np.asarray(part['reference_weights'])
    np.testing.assert_array_equal(ref,np.ones(d)/np.sqrt(d))
    old=part['candidates'][0]['model'];rank=part['ranking'];seed=sorted(rank[:int(np.ceil(d/2))])
    protected=MajorityTerms(A,labels,eps=.01,weights=ref).bounds(range(d),allowed_enemies=2)>1e-10
    proxy=NominalProxy(A,labels,ref,protected);counts=Counter()
    def evaluate(fs):
        c=_candidate(A,labels,ref,list(fs),len(fs))
        check_candidate(A,labels,ref,list(fs),c,counts)
        w=np.asarray(c['weights']);fresh=independent_loo(A,labels,list(fs),w)
        assert abs(fresh-proxy(fs)['training_LOO_BA'])<1e-12
        counts['independent_nominal_LOO_replays']+=1
        return c
    result=search(d,seed,proxy,evaluate)
    check_candidate(A,labels,ref,old['features'],old,counts)
    oldba=independent_loo(A,labels,old['features'],old['weights'])
    selected=result['selected'];gain=0.;loss=None
    if selected is not None:
        gain=(len(old['features'])-len(selected['features']))/d
        loss=selected['proxy']['training_LOO_BA']-oldba
    result.update({'dataset':name,'d':d,'outer_training_fold':row['fold'],
       'enclosing_training_indices':outer.tolist(),'inner_training_indices':fit.tolist(),
       'old_certified_count':len(old['features']),'old_training_LOO_BA':oldba,
       'additional_removal':gain,'selected_minus_old_training_LOO_BA':loss,
       'strict_count_recovery':bool(gain>0), 'source_hashes':hashes,
       'independent_check_counts':dict(counts),'complete':True,
       'seconds':time.perf_counter()-started,'new_outer_scoring':False})
    assert freeze()==hashes
    path.write_text(json.dumps(result,separators=(',',':')))
    print(name,'checks',result['exact_checks'],'recovery',gain,'train_delta',loss,flush=True)
    return result


def main():
    rows=[run(name) for name in NAMES]
    recovered=[r for r in rows if r['strict_count_recovery']]
    mean=float(np.mean([r['additional_removal'] for r in rows]))
    gate=len(recovered)>=3 and mean>=.05 and all(r['selected_minus_old_training_LOO_BA']>=-.005-1e-12 for r in recovered)
    summary={'complete':True,'datasets':len(rows),'recovered_datasets':len(recovered),
       'mean_additional_removal':mean,'resource_gate':gate,'predictive_superiority_established':False,
       'new_outer_scoring':False,'rows':[{k:r[k] for k in ('dataset','d','exact_checks',
           'old_certified_count','strict_count_recovery','additional_removal',
           'selected_minus_old_training_LOO_BA','independent_check_counts','seconds')} for r in rows]}
    (FOLDER/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
