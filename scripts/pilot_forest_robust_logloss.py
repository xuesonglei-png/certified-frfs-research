"""Three prespecified reference-only numerical robust-risk prototypes."""
import json,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_nested_joint_path import ROOT,source_verified,digest
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.fuzzy_interaction_tree import interaction_tree
from certified_frfs.orthogonal_forest_logistic import fit_orthogonal_forest
from certified_frfs.forest_robust_logloss import fit_robust_forest_logloss,all_pair_box_witnesses

FOLDER = ROOT/'results/forest-robust-logloss-pilot'; NAMES = ('iris','wine','sonar')


def envelope_objective(A,labels,fit):
    oracle = all_pair_box_witnesses(A,labels,fit,eps=.01); q = len(fit['classes']); n = len(labels)
    alpha = np.array([1/(q*sum(labels == c)) for c in labels])
    W,V = np.array(fit['coef']),np.array(fit['pair_coef']).reshape(q,len(fit['edges']),3,3)
    if q == 2: W,V = W[1],V[1]
    else: W,V = W-W.mean(axis=0,keepdims=True),V-V.mean(axis=0,keepdims=True)
    return float(alpha@oracle['envelope_losses']+(np.sum(W*W)+np.sum(V*V))/(2*n))


if __name__ == '__main__':
    threadpool_limits(1); FOLDER.mkdir(exist_ok=True); rows = []
    for name in NAMES:
        source = source_verified(name); hashes = dict(source['fingerprints'])
        for p in (Path(__file__),ROOT/'src/certified_frfs/forest_robust_logloss.py',ROOT/'research/forest-robust-logloss-pilot-protocol.md',
            ROOT/'research/forest-robust-logloss-lemma.md'): hashes[p.relative_to(ROOT).as_posix()] = digest(p)
        artifact = FOLDER/(name+'.json')
        if artifact.exists():
            saved = json.loads(artifact.read_text()); assert saved['fingerprints'] == hashes
        else:
            X,y = load_dataset(name); s = source['splits'][0]; inner = s['inner'][0]; ri,a = np.array(s['reference_indices']),np.array(inner['reference_indices'])
            raw,labels = X[ri[a]],y[ri[a]]; A,_ = preprocess_fit(raw,raw); tree = interaction_tree(A,labels); fs = list(range(A.shape[1]))
            ordinary = fit_orthogonal_forest(A,labels,fs,tree['edges'],C=1.,pair_penalty=1.); start = time.perf_counter()
            fit = fit_robust_forest_logloss(A,labels,fs,tree['edges'],C=1.,eps=.01)
            oldvalue = envelope_objective(A,labels,ordinary); newvalue = envelope_objective(A,labels,fit) if fit.get('usable') else None
            saved = {'dataset':name,'fingerprints':hashes,'outer_reference_indices':ri.tolist(),'reference_indices':a.tolist(),
                'tree':tree,'ordinary':ordinary,'fit':fit,'old_envelope_objective':oldvalue,'new_envelope_objective':newvalue,
                'seconds':time.perf_counter()-start,'archived_query_labels_used':False,'scope':'reference_only_numerical_robust_loss_pilot_not_prediction'}
            artifact.write_text(json.dumps(saved,indent=2),encoding='utf-8')
        row = {'dataset':name,'usable':saved['fit']['usable'],'status':saved['fit']['status'],'old_objective':saved['old_envelope_objective'],
            'new_objective':saved['new_envelope_objective'],'seconds':saved['seconds']}
        row['relative_improvement'] = None if not row['usable'] else (row['old_objective']-row['new_objective'])/row['old_objective']
        rows.append(row); print(row,flush=True)
    gate = all(r['usable'] and r['new_objective'] <= r['old_objective']+1e-6 for r in rows) and float(np.mean([r['relative_improvement'] for r in rows])) >= .001
    result = {'rows':rows,'numeric_resource_gate':bool(gate),'independent_audit_required':True,'goal_achieved':False,
        'archived_query_labels_used':False,'scope':'training_envelope_optimizer_mechanism_not_prediction_or_feature_selection_superiority'}
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(result,flush=True)
