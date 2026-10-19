"""Rebuild fixed-witness certificate and pair envelope; sample shared tables."""
import hashlib
import json
import zipfile
import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.majority_metric import MajorityTerms
from certified_frfs.joint_milp import pairwise_loss_envelope
from certified_frfs.core import gamma


def main():
    threadpool_limits(1); folder=ROOT/'results/joint-milp-diagnostic'; path=folder/'complete.json'; p=json.loads(path.read_text())
    for rel,expected in p['source_hashes'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==expected
    checked=0; samples=0; minoritychecks=0; records=[]
    for row in p['rows']:
        name=row['dataset']; X,y=load_dataset(name); ri=np.array(row['reference_indices']); A,_=preprocess_fit(X[ri],X[ri]); labels=y[ri]
        assert row['data_sha256']==hashlib.sha256((ROOT/'data/processed'/f'{name}.npz').read_bytes()).hexdigest()
        assert row['prior_sha256']==hashlib.sha256((ROOT/'results/guarded-support-diagnostic'/f'{name}.json').read_bytes()).hexdigest()
        w=np.ones(row['d'])/np.sqrt(row['d']); c=row['fit']; v=np.array(c['weights']); assert abs(v.sum()-w.sum())<1e-10
        assert c['features']==np.flatnonzero(v>0).tolist()
        bound=pairwise_loss_envelope(A,labels,w,v)['upper_bound']; assert bound<=.1
        assert abs(bound-c['certificate']['dependency_certificate']['upper_bound'])<1e-12
        workspace=MajorityTerms(A,labels,weights=w); protected=workspace.bounds(range(row['d']),allowed_enemies=2)>1e-10
        targets=c['fit']['targets']; assert all(protected[i] for i in targets)
        for q,r in zip(np.unique(labels),c['fit']['required']):assert sum(labels[i]==q for i in targets)>=r
        if c['full_fallback']:np.testing.assert_allclose(v,w,atol=0,rtol=0)
        else:
            for ex in c['fit']['exceptions']:
                i=ex['anchor']; enemies=np.setdiff1d(np.flatnonzero(labels!=labels[i]),ex['excluded_enemies']);assert len(ex['excluded_enemies'])<=2
                C=workspace.terms[:,i,enemies,:].transpose(1,2,0).reshape(-1,row['d'])
                assert np.all(C@v>=c['fit']['threshold']-1e-12)
        rng=np.random.default_rng(1611); maximum=-np.inf
        for _ in range(10):
            Z=np.clip(A+rng.uniform(-.01,.01,A.shape),0,1); loss=gamma(Z,labels,weights=w)-gamma(Z,labels,weights=v)
            assert loss<=bound+1e-10; maximum=max(maximum,loss)
            D=cdist(Z*v,Z*v,'cityblock');np.fill_diagonal(D,np.inf);nn=np.argsort(D,axis=1,kind='stable')[:,:5]
            assert np.all(((labels[nn]==labels[:,None]).sum(axis=1)>=3)[targets]);samples+=1;minoritychecks+=len(targets)
        records.append({'dataset':name,'bound':bound,'maximum_sampled_loss':maximum});checked+=1;print(name,'joint certificate audited',flush=True)
    result={'metrics_checked':checked,'shared_table_samples':samples,'protected_majority_checks':minoritychecks,
        'observed_violations':0,'rows':records,'result_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'scope':'constraint_and_envelope_reconstruction_sampling_not_solver_refit_or_global_optimality_or_rounding_proof'}
    (folder/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    archive=ROOT/'output/joint-milp-generation-evidence.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for rel in p['source_hashes']:z.write(ROOT/rel,rel)
        for f in [path,folder/'audit.json',ROOT/'scripts/audit_joint_milp.py',ROOT/'tests/test_joint_milp.py']:z.write(f,str(f.relative_to(ROOT)))
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)

if __name__=='__main__':main()
