"""Independent score/selection replay for robust projection v1.

Child source fitting is replayed by the existing independent hard-aware
auditor; distance/vote and new robust selection are reconstructed here.
The dependency envelope is recomputed with the production bound, so this is
not an independent reimplementation of that pre-existing analytic bound.
"""
import json
from collections import Counter
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.feasible_transport import weight_certificate
from audit_classwise_repair_admission import independent_class_certificate
from audit_hardaware_guarded_soft5_admission import independent_hard_BA
from audit_hardaware_soft5_knn_pilot import check_stage
from audit_certified_metric_knn_pool import independent_evaluate, query_certificate
from audit_forest_compact_transfer import equivalent
from run_robustaware_projection_v1 import ROOT, OUT, BASE, FREEZE, DATASETS, digest


def replay_fit(A, y, fit, counts):
    fs, anchor = fit['features'], np.asarray(fit['anchor'])
    full = np.ones(A.shape[1]) / np.sqrt(A.shape[1])
    def guard(w, saved):
        cert = weight_certificate(A, y, fs, full, w)
        classes = independent_class_certificate(y, cert)
        equivalent(cert, saved['certificate']); equivalent(classes, saved['class_certificate'])
        accepted = bool(cert['feasible'] and classes['feasible_all_classes'])
        assert accepted == saved['accepted']; counts['guards'] += 1
        return accepted
    assert fit['epsilon'] == .01 and fit['draws'] == 8
    available = guard(anchor, fit['anchor_guard'])
    if not available:
        assert fit['status'] == 'anchor_outside_original_full_certificate'
        np.testing.assert_array_equal(fit['weights'], anchor)
        assert not fit['trace']; counts['unavailable'] += 1
        return
    targets, fits = fit['candidate_targets'], fit['candidate_fits']
    learners = []
    for i, family in enumerate(('simplex_soft5_unconstrained', 'simplex_soft5_constrained', 'hardaware_soft5')):
        details = {**fits[i], 'neighborhood_k': 5}
        if i == 2: details['source_learner_ids'] = [0, 1]
        learners.append({'features': fs, 'anchor': anchor.tolist(), 'target': targets[i],
                         'fit': details, 'family': family})
    check_stage(A, y, {'learners': learners, 'certificate_checks': []}, counts)
    target = np.asarray(targets[2]); accepted = []
    for i, (t, row) in enumerate(zip((0., .05, .1, .25, .5, .75, 1.), fit['trace'], strict=True)):
        w = (1-t)*anchor+t*target
        np.testing.assert_array_equal(w, row['weights'])
        assert row['source'] == 1 and row['interpolation'] == t
        clean = independent_hard_BA(A[:, fs], y, w)
        scores = []
        for draw in range(8):
            rng = np.random.default_rng(20261011+draw)
            Z = np.clip(A[:, fs]+rng.uniform(-.01, .01, (len(A), len(fs))), 0, 1)
            scores.append(independent_hard_BA(Z, y, w))
        equivalent(clean, row['clean_BA']); equivalent(float(np.mean(scores)), row['perturbed_BA'])
        assert guard(w, row['certificate']) == row['accepted']
        counts['LOO_votes'] += 9
        if row['accepted']: accepted.append(row)
    selected = min(accepted, key=lambda r: (-round(r['perturbed_BA'],12), -round(r['clean_BA'],12),
        round(float(np.abs(np.asarray(r['weights'])-anchor).sum()),12), r['source'], r['interpolation']))
    np.testing.assert_array_equal(selected['weights'], fit['weights'])
    assert selected['interpolation'] == fit['selected_interpolation']
    counts['certified_selected'] += 1


def replay_evaluation(A, y, B, truth, fs, weights):
    report = independent_evaluate(A,y,B,truth,fs,weights)
    cert = query_certificate(A,y,B,np.asarray(report['predictions']),fs,weights)
    good = (np.asarray(report['predictions']) == truth) & np.asarray(cert['certified'])
    report['metrics']['query_certified_BA'] = float(np.mean([np.mean(good[truth==c]) for c in np.unique(truth)]))
    perturbations = []
    for draw in range(16):
        rng = np.random.default_rng(20261060+draw)
        Z = np.clip(A+rng.uniform(-.01,.01,A.shape),0,1)
        Q = np.clip(B+rng.uniform(-.01,.01,B.shape),0,1)
        perturbations.append(independent_evaluate(Z,y,Q,truth,fs,weights)['metrics']['clean_BA'])
    report['metrics']['mean_perturbed_BA'] = float(np.mean(perturbations))
    return report['metrics']


def main():
    threadpool_limits(1)
    freeze = json.loads(FREEZE.read_text())
    for rel, sha in freeze['fingerprints'].items(): assert digest(ROOT/rel) == sha, rel
    summary = json.loads((OUT/'complete.json').read_text()); counts = Counter()
    for name in DATASETS:
        row = json.loads((OUT/(name+'.json')).read_text())
        assert row['error'] is None and not row['heldout_labels_used_for_selection']
        assert row['source_sha256'] == digest(BASE/(name+'.json'))
        X,y=load_dataset(name);ri=np.asarray(row['reference_indices']);vi=np.asarray(row['validation_indices'])
        assert not set(ri)&set(vi); A,B=preprocess_fit(X[ri],X[vi])
        replay_fit(A,y[ri],row['fit'],counts)
        for arm,w in (('baseline',row['anchor']),('proposed',row['selected_weights'])):
            metrics = replay_evaluation(A,y[ri],B,y[vi],row['features'],np.asarray(w))
            equivalent(metrics,row[arm+'_metrics']); counts['outer_predictions'] += 1
        counts['datasets'] += 1; print(name,'independent replay PASS',flush=True)
    report={'passed':True,'counts':dict(counts),'summary_sha256':digest(OUT/'complete.json'),
        'artifact_hashes':{name:digest(OUT/(name+'.json')) for name in DATASETS},
        'auditor_sha256':digest(Path(__file__)),'goal_achieved':False,
        'scope':'independent_votes_selection_child_source_audit_production_dependency_envelope_recomputed'}
    (OUT/'complete-audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report['counts'],indent=2),flush=True)


if __name__=='__main__': main()
