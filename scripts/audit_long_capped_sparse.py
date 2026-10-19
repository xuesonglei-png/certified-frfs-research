"""Rebuild all long-trajectory mixture/sparse candidates and outer scores."""
import json
import zipfile
import hashlib
import numpy as np
from threadpoolctl import threadpool_limits
from diagnose_long_capped_sparse import ROOT,FOLDER
from audit_trajectory_recent import scaling
from certified_frfs.datasets import load_dataset
from certified_frfs.query_probability import QueryProbabilityObjective
from certified_frfs.trajectory_certificate import certify_capped_trajectory
from certified_frfs.capped_sparse import sparse_trajectory_extension
from certified_frfs.validated_pool import validation_scores


def main():
    threadpool_limits(1);counts={'splits':0,'trajectory_values':0,'rebuilt_candidates':0,'external_scores':0}
    for path in sorted(FOLDER.glob('*.json')):
        if path.name.startswith('complete'):continue
        p=json.loads(path.read_text());name=p['dataset']
        for relative, expected in p['fingerprints'].items():
            assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==expected, relative
        prior=json.loads((ROOT/'results/capped-sparse-extension'/f'{name}.json').read_text());X,y=load_dataset(name)
        for index,s in enumerate(p['splits']):
            old=prior['splits'][index];ri,vi=np.array(s['reference_indices']),np.array(s['validation_indices'])
            assert ri.tolist()==old['reference_indices'] and vi.tolist()==old['validation_indices'] and set(ri).isdisjoint(vi)
            trajectory=s['trajectory'];obj=QueryProbabilityObjective(X[ri],y[ri]);assert obj.splits==trajectory['training_splits']
            assert trajectory['steps']==10 and trajectory['oracle_seconds']==2 and trajectory['weight_cap_ratio']==2 and not trajectory['certified']
            full=np.ones(p['d'])/np.sqrt(p['d']);mass=full.sum();last=np.inf
            for node in trajectory['trace']:
                v=np.array(node['weights']);value=obj(v/mass)[0]
                assert v.min()>=0 and v.max()<=2*mass/p['d']+1e-10 and abs(v.sum()-mass)<1e-10
                assert abs(value-node['objective'])<1e-10 and value<=last+1e-12
                last=value;counts['trajectory_values']+=1
            rebuilt=certify_capped_trajectory(X[ri],y[ri],trajectory)
            assert rebuilt==s['certified_trajectory']
            fit=sparse_trajectory_extension(X[ri],y[ri],rebuilt)
            assert fit==s['fit'];counts['rebuilt_candidates']+=len(fit['candidates'])
            A,B=scaling(X[ri],X[vi]);c=fit['compact_selected']
            metrics=validation_scores(A,y[ri],B,y[vi],c['features'],np.array(c['weights']),144+index,joint=True)
            assert all(abs(metrics[k]-s['metrics'][k])<1e-12 for k in metrics)
            counts['splits']+=1;counts['external_scores']+=2
        print(name,'long pipeline audit complete',flush=True)
    result={**counts,'scope':'trajectory_loss_checks_full_candidate_refits_certification_selection_scores_not_LP_optimizer_refit_or_rounding_proof'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with zipfile.ZipFile(ROOT/'output/long-capped-sparse-evidence.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in FOLDER.glob('*.json'):archive.write(path,str(path.relative_to(ROOT)))
        for path in (ROOT/'src/certified_frfs').glob('*.py'):archive.write(path,str(path.relative_to(ROOT)))
        for rel in ('scripts/diagnose_long_capped_sparse.py','scripts/audit_long_capped_sparse.py','research/long-capped-sparse-protocol.md'):
            archive.write(ROOT/rel,rel)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
