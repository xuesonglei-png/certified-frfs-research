"""Recompute valid witness simplex objectives and shared-table loss samples."""
import json,hashlib,zipfile
import numpy as np
from threadpoolctl import threadpool_limits
from diagnose_witness_mixture import ROOT,FOLDER,tasks
from audit_trajectory_recent import scaling
from certified_frfs.datasets import load_dataset
from certified_frfs.witness_mixture import coordinate_envelopes
from certified_frfs.redistributed_loss import redistributed_loss_bound
from certified_frfs.core import interval_bounds,gamma,_interval_distances


def main():
    threadpool_limits(1);counts={'candidates':0,'simplex_envelopes':0,'sampled_shared_losses':0}
    for task in tasks():
        path=FOLDER/f"{task['group']}-{task['dataset']}-{task['split']}-{task['candidate']}.json"
        p=json.loads(path.read_text());assert p['task']==task
        for relative,value in p['fingerprints'].items():assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==value,relative
        prior_path=ROOT/task['source'];assert hashlib.sha256(prior_path.read_bytes()).hexdigest()==p['prior_sha256']
        source=json.loads(prior_path.read_text());s=source['splits'][task['split']];old=s['fit']['candidates'][task['candidate']]
        if task['group']=='original15':X,y=load_dataset(task['dataset']);ri=np.array(s['reference_indices']);dp=ROOT/'data/processed'/f"{task['dataset']}.npz"
        else:
            dp=ROOT/'data/newdata-compact'/f"{task['dataset']}.npz"
            with np.load(dp) as data:X,y=data['X'],data['y']
            ri=np.array(s['train_indices'])
        assert hashlib.sha256(dp.read_bytes()).hexdigest()==p['data_sha256'] and ri.tolist()==p['reference_indices']
        A,_=scaling(X[ri],X[ri]);labels=y[ri];fs=p['features'];v=np.array(p['weights']);full=np.ones(X.shape[1])/np.sqrt(X.shape[1])
        assert old['weights']==p['weights'] and old['features']==fs and old['certificate']['majority_feasible']
        base=redistributed_loss_bound(A,labels,fs,full,v[fs]);result=p['result'];assert abs(base['upper_bound']-p['old_upper'])<1e-12
        lo,hi=interval_bounds(A,.01);lb,ub=_interval_distances(lo,hi,fs,v)
        by_anchor={}
        for check in result['mixture_checks']:
            i,j=check['anchor'],check['enemy'];witnesses=np.array(check['witnesses'])
            grids,knots=coordinate_envelopes(lo,hi,i,j,witnesses,full,v)
            assert knots==check['knots'];lam=np.array(check['fit']['lambda'])
            assert lam.min()>=0 and abs(lam.sum()-1)<1e-12
            value=float(sum(np.max(lam@g) for g in grids))
            vertices=np.array([sum(np.max(g[k]) for g in grids) for k in range(len(lam))])
            assert abs(value-check['fit']['upper'])<1e-10 and value<=vertices.min()+1e-10
            bound=max(0.,min(1-lb[i,j],value));assert abs(bound-check['bound'])<1e-10
            by_anchor.setdefault(i,[]).append((j,bound));counts['simplex_envelopes']+=1
        values=[]
        for i in range(len(labels)):
            enemies=np.flatnonzero(labels!=labels[i]);possible=enemies[lb[i,enemies]<=np.min(ub[i,enemies])+1e-12]
            expected={int(j) for j in possible if lb[i,j]<1}
            if base['redistributed_object_bounds'][i]==0:values.append(0.);continue
            actual={j for j,_ in by_anchor.get(i,[])};assert actual==expected
            bounds=[b for _,b in by_anchor.get(i,[])]+[0.]
            values.append(min(max(bounds),base['redistributed_object_bounds'][i]))
        raw=float(np.mean(values));upper=min(base['upper_bound'],float(np.clip(raw+1e-10*(1+abs(raw)),0,1)))
        assert np.allclose(values,result['mixture_object_bounds'],atol=1e-12,rtol=0)
        assert abs(upper-result['upper_bound'])<1e-12 and p['admitted']==(upper<=.1)
        rng=np.random.default_rng(17333)
        for _ in range(3):
            Z=np.clip(A+rng.uniform(-.01,.01,A.shape),0,1)
            assert gamma(Z,labels,weights=full)-gamma(Z,labels,fs,weights=v)<=upper+1e-10
            counts['sampled_shared_losses']+=1
        counts['candidates']+=1;print(task['dataset'],task['split'],task['candidate'],'mixture audit complete',flush=True)
    result={**counts,'scope':'simplex_objective_bound_aggregation_shared_samples_not_LP_optimality_or_directed_rounding_proof'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with zipfile.ZipFile(ROOT/'output/witness-mixture-evidence.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in FOLDER.glob('*.json'):archive.write(path,str(path.relative_to(ROOT)))
        for path in (ROOT/'src/certified_frfs').glob('*.py'):archive.write(path,str(path.relative_to(ROOT)))
        for rel in ('scripts/diagnose_witness_mixture.py','scripts/audit_witness_mixture.py','research/witness-mixture-protocol.md','tests/test_witness_mixture.py'):
            archive.write(ROOT/rel,rel)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
