"""ALL seven source failures, training-only budget diagnostic."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_simplex_soft5_knn_pilot import ROOT, FOLDER, digest
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.softvote_metric import softvote_terms
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.classwise_dependency_certificate import classwise_dependency_certificate
from simplex_soft_topk_targets import fit_simplex_target
from soft_topk_predictive_metric import anchor_witness_polytope
from soft_topk_metric_objective import soft_topk_objective
from hardaware_guarded_soft5_descent import hard5_LOO_BA


def main():
    threadpool_limits(1)
    inventory = ROOT/'research/simplex-soft5-source-optimizer-failures.json'
    failures = json.loads(inventory.read_text())['failures']
    assert len(failures) == 7
    sources = [Path(__file__), inventory, ROOT/'research/simplex-source-budget-diagnostic-protocol.md',
               ROOT/'scripts/simplex_soft_topk_targets.py', ROOT/'scripts/soft_topk_metric_objective.py',
               ROOT/'scripts/soft_topk_predictive_metric.py', ROOT/'scripts/hardaware_guarded_soft5_descent.py']
    sources += sorted((ROOT/'src/certified_frfs').glob('*.py'))
    for name in sorted({r['dataset'] for r in failures}):
        sources += [FOLDER/(name+'.json'), ROOT/'data/processed'/(name+'.npz')]
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in sources}
    rows = []
    for item in failures:
        name = item['dataset']; saved = json.loads((FOLDER/(name+'.json')).read_text())
        block = saved['inner'][item['block']]; source = block['learners'][item['learner']]
        assert source['fit']['status'] == 'optimizer_failed_anchor_fallback'
        assert source['fit']['optimizer_iterations'] == 200
        X, y = load_dataset(name); ri = np.array(saved['reference_indices'])
        ca = np.array(block['reference_indices'])
        raw = X[ri][ca]; labels = y[ri][ca]
        # No outer or inner validation table is read for this diagnostic.
        # Shared sklearn transformer needs a nonempty dummy query; use a
        # training row and discard it, never a validation observation.
        A, _ = preprocess_fit(raw, raw[:1])
        fs = source['features']; anchor = np.array(source['anchor']); constrained = source['fit']['constrained']
        assert fs == item['features'] and source['family'] == item['family']
        target, fit = fit_simplex_target(A, labels, fs, anchor, constrained=constrained, max_iter=1000)
        mass = np.sqrt(A.shape[1]); terms = softvote_terms(A[:,fs], labels, eps=.01)
        initial = soft_topk_objective(anchor/mass, terms, .05, .01)[0]
        final = soft_topk_objective(target/mass, terms, .05, .01)[0]
        C, rhs, _ = anchor_witness_polytope(A, labels, fs, anchor)
        slack = C@target-rhs
        cert = weight_certificate(A, labels, fs, np.ones(A.shape[1])/mass, target)
        classes = classwise_dependency_certificate(labels, cert)
        assert np.isfinite(target).all() and target.min() >= 0 and abs(target.sum()-mass) <= 1e-8
        if fit['status'] == 'local_simplex_soft5_target':
            assert fit['optimizer_success'] and final <= initial+1e-8
            assert abs(final-fit['training_objective_final']) < 1e-10
            if constrained: assert np.all(slack >= 0)
        row = {**item, 'retry_fit': fit, 'objective_initial_recomputed': initial,
               'objective_final_recomputed': final,
               'minimum_anchor_witness_slack': float(slack.min()) if len(slack) else None,
               'original_full_guard': {'certificate': cert, 'class_certificate': classes,
                   'accepted': bool(cert['feasible'] and classes['feasible_all_classes'])},
               'hard_training_LOO_BA_anchor': hard5_LOO_BA(A[:,fs], labels, anchor),
               'hard_training_LOO_BA_retry': hard5_LOO_BA(A[:,fs], labels, target)}
        rows.append(row)
        print(name, item['block'], item['learner'], fit['status'], fit['optimizer_iterations'],
              'hard_delta_pp', 100*(row['hard_training_LOO_BA_retry']-row['hard_training_LOO_BA_anchor']), flush=True)
    assert hashes == {p.relative_to(ROOT).as_posix(): digest(p) for p in sources}
    result = {'rows': rows, 'fingerprints': hashes, 'complete': True, 'goal_achieved': False,
              'query_labels_used': False, 'scope': 'all_seven_failed_sources_training_only_budget_diagnostic',
              'successful_retries': sum(r['retry_fit']['status'] == 'local_simplex_soft5_target' for r in rows),
              'original_full_guard_accepted': sum(r['original_full_guard']['accepted'] for r in rows)}
    (ROOT/'research/simplex-source-budget-diagnostic.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print({k: v for k, v in result.items() if k not in ('rows','fingerprints')}, flush=True)


if __name__ == '__main__': main()
