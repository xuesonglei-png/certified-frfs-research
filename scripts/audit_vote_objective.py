"""Reconstruct soft-five loss traces, inclusion mass and held-out predictions."""
import hashlib
import json
import zipfile
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from audit_sparse_pilots import preprocessing
from certified_frfs.datasets import load_dataset
from certified_frfs.vote_objective import VoteMassObjective, soft_neighbor_inclusion
from certified_frfs.validated_pool import validation_scores


def main():
    threadpool_limits(1);folder=ROOT/'results/vote-objective-diagnostic';path=folder/'complete.json';p=json.loads(path.read_text())
    for rel,expected in p['source_hashes'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==expected
    prior=ROOT/'results/feasible-query-ablation/complete.json';assert hashlib.sha256(prior.read_bytes()).hexdigest()==p['prior_sha256']
    old=json.loads(prior.read_text());counts={'trace_values':0,'inclusion_query_checks':0,'external_scores':0,'splits':0}
    for row in p['rows']:
        X,y=load_dataset(row['dataset']);past=next(r for r in old['rows'] if r['dataset']==row['dataset'])
        for index,s in enumerate(row['splits']):
            assert s['reference_indices']==past['splits'][index]['reference_indices'];assert s['validation_indices']==past['splits'][index]['validation_indices']
            ri,vi=np.array(s['reference_indices']),np.array(s['validation_indices']);assert not set(ri)&set(vi)
            A,B=preprocessing(X[ri],X[vi]);objective=VoteMassObjective(X[ri],y[ri]);fit=s['fit'];assert objective.splits==fit['training_splits'];assert fit['certificate'] is None
            before=np.inf;mass=np.sqrt(row['d'])
            for node in fit['trace']:
                w=np.array(node['weights']);assert np.all(w>=0) and abs(w.sum()-mass)<1e-10
                value,_=objective(w/mass);assert abs(value-node['objective'])<1e-10 and value<before;before=value;counts['trace_values']+=1
                for C,_,_ in objective.terms:
                    distance=np.einsum('ijd,d->ij',C,w/mass);k=min(5,distance.shape[1]);inc=soft_neighbor_inclusion(distance,k=k)
                    np.testing.assert_allclose(inc.sum(axis=1),k,rtol=0,atol=1e-11);counts['inclusion_query_checks']+=len(distance)
            np.testing.assert_array_equal(fit['weights'],fit['trace'][-1]['weights'])
            scores=validation_scores(A,y[ri],B,y[vi],fit['features'],np.array(fit['weights']),144+index,joint=True)
            assert all(abs(scores[k]-s['metrics'][k])<1e-12 for k in scores);counts['external_scores']+=2;counts['splits']+=1
        print(row['dataset'],'softfive audited',flush=True)
    result={**counts,'result_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'scope':'loss_inclusion_score_rebuild_not_optimizer_refit_or_risk_certificate'}
    (folder/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with zipfile.ZipFile(ROOT/'output/vote-objective-evidence.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for rel in p['source_hashes']:z.write(ROOT/rel,rel)
        for f in folder.glob('*.json'):z.write(f,str(f.relative_to(ROOT)))
        z.write(ROOT/'scripts/audit_vote_objective.py','scripts/audit_vote_objective.py')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
