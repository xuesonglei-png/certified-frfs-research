"""Fixed ratio2 coordinatecap probability descent on all44 internal splits."""
import hashlib
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit, validation_scores
from certified_frfs.feasible_query_descent import fit_feasible_query


def main():
    threadpool_limits(1); folder=ROOT/'results/capped-query-diagnostic';folder.mkdir(exist_ok=True);out=folder/'complete.json'
    if out.exists():raise FileExistsError(out)
    source=ROOT/'results/feasible-query-ablation/complete.json';prior=json.loads(source.read_text());rows=[]
    paths=sorted((ROOT/'src/certified_frfs').glob('*.py'))+[Path(__file__),ROOT/'research/capped-query-protocol.md']
    hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths};utility=lambda m:(m['clean']+m['noise01'])/2
    for old in prior['rows']:
        X,y=load_dataset(old['dataset']);splits=[]
        for index,s in enumerate(old['splits']):
            ri,vi=np.array(s['reference_indices']),np.array(s['validation_indices']);A,B=preprocess_fit(X[ri],X[vi])
            fit=fit_feasible_query(X[ri],y[ri],certified=False,oracle_seconds=10.,weight_cap_ratio=2.)
            assert fit['training_splits']==s['fit']['training_splits']
            scores=validation_scores(A,y[ri],B,y[vi],fit['features'],np.array(fit['weights']),144+index,joint=True)
            splits.append({'reference_indices':ri.tolist(),'validation_indices':vi.tolist(),'fit':fit,'metrics':scores})
        means={'capped':float(np.mean([utility(s['metrics']) for s in splits])),**old['means']}
        rows.append({'dataset':old['dataset'],'d':old['d'],'splits':splits,'means':means});print(old['dataset'],means,flush=True)
    means={m:float(np.mean([r['means'][m] for r in rows])) for m in rows[0]['means']};positive=sum(r['means']['capped']>r['means']['full']+1e-12 for r in rows)
    gates={'mean_gain_full_half_point':means['capped']-means['full']>=.005,'ten_positive_full':positive>=10,'not_below_certified_mean':means['capped']>=means['certified']-1e-12}
    assert hashes=={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths}
    result={'rows':rows,'means':means,'positive_full':positive,'gates':gates,'gate_passed':all(gates.values()),'source_hashes':hashes,'prior_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'scope':'fixedcap_unconstrained_all_inner_development_not_certification_or_test_superiority'}
    out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k not in ('rows','source_hashes')},indent=2),flush=True)

if __name__=='__main__':main()
