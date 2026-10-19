import json,hashlib,zipfile
import numpy as np
from threadpoolctl import threadpool_limits
from diagnose_certified_query_refinement import ROOT,FOLDER
from certified_frfs.datasets import load_dataset
from certified_frfs.certified_query_refinement import refine_certified_query
from certified_frfs.validated_pool import preprocess_fit,validation_scores


def main():
    threadpool_limits(1);counts={'splits':0,'accepted_updates':0,'replayed_attempts':0,'external_scores':0}
    for path in sorted(FOLDER.glob('*.json')):
        if path.name.startswith('complete'):continue
        p=json.loads(path.read_text());name=p['dataset']
        for relative,value in p['fingerprints'].items():assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==value,relative
        prior=json.loads((ROOT/'results/long-capped-sparse'/path.name).read_text());X,y=load_dataset(name)
        for index,s in enumerate(p['splits']):
            old=prior['splits'][index];ri,vi=np.array(s['reference_indices']),np.array(s['validation_indices'])
            assert ri.tolist()==old['reference_indices'] and vi.tolist()==old['validation_indices'] and set(ri).isdisjoint(vi)
            initial=old['fit']['compact_selected'];fit=refine_certified_query(X[ri],y[ri],initial)
            assert fit==s['fit'] and set(fit['features'])<=set(initial['features'])
            assert all(node['certificate']['feasible'] for node in fit['trace'])
            assert np.all(np.diff([node['objective'] for node in fit['trace']])<0)
            A,B=preprocess_fit(X[ri],X[vi]);metrics=validation_scores(A,y[ri],B,y[vi],fit['features'],np.array(fit['weights']),144+index,joint=True)
            assert all(abs(metrics[k]-s['metrics'][k])<1e-12 for k in metrics)
            assert s['initial_metrics']==old['metrics']
            counts['splits']+=1;counts['accepted_updates']+=len(fit['trace'])-1
            counts['replayed_attempts']+=len(fit['attempts']);counts['external_scores']+=2
        print(name,'certified refinement replay audited',flush=True)
    result={**counts,'scope':'full_optimizer_replay_certificates_objectives_external_scores_not_directed_rounding_or_independent_mathematical_proof'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with zipfile.ZipFile(ROOT/'output/certified-query-refinement-evidence.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in FOLDER.glob('*.json'):archive.write(path,str(path.relative_to(ROOT)))
        for path in (ROOT/'src/certified_frfs').glob('*.py'):archive.write(path,str(path.relative_to(ROOT)))
        for rel in ('scripts/diagnose_certified_query_refinement.py','scripts/audit_certified_query_refinement.py','research/certified-query-refinement-protocol.md','tests/test_certified_query_refinement.py'):
            archive.write(ROOT/rel,rel)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
