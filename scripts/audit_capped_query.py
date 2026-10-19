"""Rebuild capped oracle fits, trace objectives and external scores."""
import hashlib
import json
import zipfile
import numpy as np
from scipy.optimize import linprog
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from audit_sparse_pilots import preprocessing
from certified_frfs.datasets import load_dataset
from certified_frfs.query_probability import QueryProbabilityObjective
from certified_frfs.validated_pool import validation_scores


def main():
    threadpool_limits(1);folder=ROOT/'results/capped-query-diagnostic';path=folder/'complete.json';p=json.loads(path.read_text());counts={'trace_values':0,'oracles_refit':0,'external_scores':0,'splits':0}
    for rel,h in p['source_hashes'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h
    source=ROOT/'results/feasible-query-ablation/complete.json';assert hashlib.sha256(source.read_bytes()).hexdigest()==p['prior_sha256'];prior=json.loads(source.read_text())
    for row in p['rows']:
        X,y=load_dataset(row['dataset']);old=next(r for r in prior['rows'] if r['dataset']==row['dataset']);d=row['d'];mass=np.sqrt(d);full=np.ones(d)/np.sqrt(d);cap=2*mass/d
        for index,s in enumerate(row['splits']):
            assert s['reference_indices']==old['splits'][index]['reference_indices'];assert s['validation_indices']==old['splits'][index]['validation_indices']
            ri,vi=np.array(s['reference_indices']),np.array(s['validation_indices']);assert not set(ri)&set(vi);A,B=preprocessing(X[ri],X[vi]);fit=s['fit'];objective=QueryProbabilityObjective(X[ri],y[ri]);assert objective.splits==fit['training_splits'];assert fit['certificate'] is None
            previous=np.inf
            for node in fit['trace']:
                w=np.array(node['weights']);assert np.all(w>=0) and np.all(w<=cap+1e-10) and abs(w.sum()-mass)<1e-10
                assert 1/np.sum((w/mass)**2)>=d/2-1e-10
                value=objective(w/mass)[0];assert abs(value-node['objective'])<1e-10 and value<previous;previous=value;counts['trace_values']+=1
            for oracle in fit['oracles']:
                costs=np.array(oracle['fit']['support_costs']);result=linprog(np.r_[costs/mass,np.full(d,.001*costs.min()/mass)],A_ub=np.column_stack([-np.eye(d),-np.eye(d)]),b_ub=-full,A_eq=np.array([np.r_[np.ones(d),np.zeros(d)]]),b_eq=[mass],bounds=[(0.,min(mass,cap))]*d+[(0.,mass)]*d,method='highs')
                assert result.success;np.testing.assert_allclose(result.x[:d],oracle['weights'],atol=1e-10,rtol=0);counts['oracles_refit']+=1
            np.testing.assert_array_equal(fit['weights'],fit['trace'][-1]['weights']);scores=validation_scores(A,y[ri],B,y[vi],fit['features'],np.array(fit['weights']),144+index,joint=True)
            assert all(abs(scores[k]-s['metrics'][k])<1e-12 for k in scores);counts['external_scores']+=2;counts['splits']+=1
        print(row['dataset'],'capped audited',flush=True)
    result={**counts,'result_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'scope':'LP_refits_loss_cap_score_rebuild_not_entire_descent_replay_or_confirmation'}
    (folder/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with zipfile.ZipFile(ROOT/'output/capped-query-evidence.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for rel in p['source_hashes']:z.write(ROOT/rel,rel)
        for f in folder.glob('*.json'):z.write(f,str(f.relative_to(ROOT)))
        z.write(ROOT/'scripts/audit_capped_query.py','scripts/audit_capped_query.py')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
