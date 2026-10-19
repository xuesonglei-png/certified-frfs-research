"""Rebuild weighted continuous screening metrics and certificates."""
import hashlib
import argparse
import json
import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.joint_milp import joint_certified_metric, pairwise_loss_envelope
from certified_frfs.majority_metric import MajorityTerms
from certified_frfs.core import gamma
from certified_frfs import baselines


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-subdir',default='weighted-continuous-diagnostic');args=parser.parse_args()
    if args.output_subdir not in ('weighted-continuous-diagnostic','weighted-continuous-slack-diagnostic'):raise ValueError('unknown diagnostic directory')
    threadpool_limits(1); folder=ROOT/'results'/args.output_subdir; path=folder/'complete.json'; p=json.loads(path.read_text()); samples=0; violations=0
    for row in p['rows']:
        name=row['dataset']; prior=ROOT/'results/guarded-support-diagnostic'/f'{name}.json'; X,y=load_dataset(name); s=json.loads(prior.read_text())['splits'][0];ri=np.array(s['reference_indices']);A,_=preprocess_fit(X[ri],X[ri]); scores=baselines.relieff_scores(A,y[ri],random_state=101); order=np.argsort(-scores,kind='stable');rank=np.empty(len(order),int);rank[order]=np.arange(len(order));costs=1+rank/max(1,len(order)-1);fit=row['fit']
        np.testing.assert_allclose(costs,row['support_costs'],rtol=0,atol=0)
        assert fit['features']==row['fit']['features'];np.testing.assert_allclose(fit['weights'],row['fit']['weights'],atol=1e-12,rtol=0);assert fit['full_fallback']==row['fit']['full_fallback']
        w=np.ones(row['d'])/np.sqrt(row['d']);v=np.array(row['fit']['weights']); bound=pairwise_loss_envelope(A,y[ri],w,v)['upper_bound'];assert bound<=.1
        C=MajorityTerms(A,y[ri],weights=w);protected=C.bounds(range(row['d']),allowed_enemies=2)>1e-10
        if not row['fit']['full_fallback']:
            for i in fit['fit']['targets']:
                enemies=np.flatnonzero(y[ri]!=y[ri][i]);ex=next(e for e in fit['fit']['exceptions'] if e['anchor']==i);retained=np.setdiff1d(enemies,ex['excluded_enemies']);coeff=C.terms[:,i,retained,:].transpose(1,2,0).reshape(-1,row['d']);assert np.all(coeff@v>=fit['fit']['threshold']-1e-12)
        rng=np.random.default_rng(1711)
        for _ in range(10):
            Z=np.clip(A+rng.uniform(-.01,.01,A.shape),0,1);assert gamma(Z,y[ri],weights=w)-gamma(Z,y[ri],weights=v)<=bound+1e-10;samples+=1
        print(name,'weighted audit',flush=True)
    result={'datasets':len(p['rows']),'samples':samples,'observed_violations':violations,'result_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'scope':'weighted_continuous_rebuild_not_test_risk_or_rounding_proof'}; (folder/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
