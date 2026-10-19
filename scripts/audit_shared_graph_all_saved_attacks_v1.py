"""All 30 saved attack tables versus both graph iterations, including caches."""
from collections import Counter
import hashlib,json
from pathlib import Path
import sys
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from audit_classwise_shared_attack_all_rows_v1 import scalar_memberships
NAMES=('banknote','glass','heart_statlog','ionosphere','seeds','wdbc')


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(version):
    folder=ROOT/('results/shared-graph-dependency-v'+str(version));tally=Counter();details=[]
    summary=json.loads((folder/'summary.json').read_text());freeze=json.loads((folder/'source-freeze.json').read_text())['hashes']
    for rel,sha in freeze.items():assert digest(ROOT/rel)==sha
    for name in NAMES:
        path=folder/(name+'.json');assert digest(path)==summary['artifact_hashes'][name]
        saved=json.loads(path.read_text());attack=json.loads((ROOT/'results/classwise-shared-attack-v1'/(name+'.json')).read_text())
        parent=json.loads((ROOT/'results/repair-relation-selector-v4'/(name+'.json')).read_text());panel=parent['rows'][0]
        ids=np.asarray(panel['train_indices'])[np.asarray(panel['fit']['inner'][0]['fit_indices'])]
        X,y=load_dataset(name);A,_=preprocess_fit(X[ids],X[ids]);y=y[ids];classes=np.unique(y)
        for ci,(row,source) in enumerate(zip(saved['rows'],attack['reports'],strict=True)):
            wide=np.zeros(A.shape[1]);wide[row['features']]=row['weights']
            for previous in source['attacks']:
                wf=np.asarray(previous['full_weights']);fit=previous['fit'];label=fit['target_class']
                index=int(np.flatnonzero(classes==label)[0]);Z=np.asarray(fit['perturbed_table'])
                assert Z.shape==A.shape and np.isfinite(Z).all()
                assert np.all(Z>=np.maximum(0.,A-.01)-1e-12) and np.all(Z<=np.minimum(1.,A+.01)+1e-12)
                actual=scalar_memberships(Z,y,range(A.shape[1]),wf)-scalar_memberships(Z,y,range(A.shape[1]),wide)
                loss=float(actual[y==label].mean())
                np.testing.assert_allclose(loss,fit['observed_signed_class_loss_lower'],atol=1e-10,rtol=0)
                matches=[g for g in row['guards'] if np.array_equal(g['full_weights'],wf)];assert matches
                for guard in matches:assert loss<=guard['new_class_bounds'][index]+1e-7
                violates=loss>.1+1e-10;assert fit['class_violation_witness']==violates
                if violates:assert not row['new_dependency_feasible'] and not row['new_joint_feasible']
                tally['all_saved_attack_tables']+=1;tally['violating_target_tables']+=violates
                tally['matched_reference_class_guards']+=len(matches)
                tally['tables_with_cached_matching_guard']+=any(g['reused_identical_metric'] for g in matches)
                details.append({'dataset':name,'candidate':ci,'class':label,'old_attack_loss':loss,
                    'graph_class_uppers':[g['new_class_bounds'][index] for g in matches],
                    'known_violation':violates})
    assert tally['all_saved_attack_tables']==30 and tally['violating_target_tables']==10
    result={'passed':True,'version':version,'checks':dict(tally),'details':details,'auditor_sha256':digest(__file__),
        'new_fit_or_attack_or_solver':False,'scope':'every_saved_attack_loss_and_both_reference_class_bounds_including_reused_guards',
        'limits':['This supplemental check does not re-embed reused graph models or independently prove solver dual numbers.',
                  'The separate sparse model audit embeds 25 matched old tables and every unique solver endpoint.',
                  'No classification accuracy or out-of-sample robustness claim.']}
    out=folder/'all-saved-attack-consistency-independent-v1.json';assert not out.exists();out.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='details'},indent=2))


if __name__=='__main__':
    threadpool_limits(1)
    for version in (1,2):run(version)
