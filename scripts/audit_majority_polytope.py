"""Refit LP and sample every returned metric, including loss-rejected ones."""
import hashlib
import json
import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from diagnose_guarded_support import FOLDER
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.majority_polytope import fit_majority_polytope
from certified_frfs.core import gamma


def main():
    threadpool_limits(1); path=FOLDER/'polytope-admission.json'; p=json.loads(path.read_text()); fits=0; samples=0
    for r in p['rows']:
        prior=FOLDER/f"{r['dataset']}.json"
        assert hashlib.sha256(prior.read_bytes()).hexdigest()==r['prior_sha256']
        s=json.loads(prior.read_text())['splits'][r['inner']]; X,y=load_dataset(r['dataset']); ri=np.array(s['reference_indices'])
        A,_=preprocess_fit(X[ri],X[ri]); labels=y[ri]; w=np.ones(A.shape[1])/np.sqrt(A.shape[1]); fs=r['features']
        v,info=fit_majority_polytope(A,labels,fs,w)
        assert info['status']==r['status']; assert info.get('certificate')==r['certificate']; fits+=1
        if v is None: continue
        np.testing.assert_allclose(v,r['weights'],rtol=0,atol=1e-12); wide=np.zeros(A.shape[1]); wide[fs]=v
        rng=np.random.default_rng(1511)
        for _ in range(30):
            Z=np.clip(A+rng.uniform(-.01,.01,A.shape),0,1)
            D=cdist(Z[:,fs]*v,Z[:,fs]*v,'cityblock'); np.fill_diagonal(D,np.inf)
            nn=np.argsort(D,axis=1,kind='stable')[:,:5]
            assert np.all(((labels[nn]==labels[:,None]).sum(axis=1)>=3)[info['targets']])
            loss=gamma(Z,labels,weights=w)-gamma(Z,labels,fs,weights=wide)
            assert loss<=info['certificate']['dependency_certificate']['upper_bound']+1e-10
            samples+=1
    sources=[ROOT/'src/certified_frfs/majority_polytope.py',ROOT/'scripts/diagnose_majority_polytope.py',ROOT/'scripts/audit_majority_polytope.py',ROOT/'research/majority-polytope-protocol.md']
    result={'refitted_supports':fits,'sampled_tables':samples,'observed_violations':0,
        'result_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'source_hashes':{str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest() for q in sources},
        'provenance_note':'generation source_sha256 hashed the script pathname, not its content; actual current content hashes are here. No independent frozen generation-source claim.',
        'scope':'LP_refitting_and_sampled_target_majority_loss_not_directed_rounding_or_test_risk'}
    (FOLDER/'polytope-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
