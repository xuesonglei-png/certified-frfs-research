"""Matched strict-solver residual-cell experiment, ALL15 training only."""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_simplex_soft5_knn_pilot import ROOT, digest
from continue_soft_topk_knn_pilot import verify
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from fit_strict_one_sided_hinge import fit_strict_one_sided_target
from fit_pair_transition_frs_hinge import fit_pair_transition_target
from frs_pair_transition_cell import build_class_cell, evaluate_class_cell as pair_evaluation
from frs_one_sided_cell import evaluate_class_cell as monotone_evaluation

SOURCE = ROOT/'results/frs-one-sided-training'
FOLDER = ROOT/'results/pair-transition-frs-training'
FREEZE = ROOT/'research/pair-transition-frs-training-source-freeze.json'
ARMS = ('one_sided', 'pair_transition')
KEYS = ('objective_decrease', 'clipped_training_bound_decrease_pp', 'pair_bound_tightening_at_old_target_pp')


def builtin(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def frozen():
    hashes = json.loads(FREEZE.read_text())['fingerprints']
    for rel, sha in hashes.items():
        assert digest(ROOT/rel) == sha, rel
    return hashes


def prerequisite():
    audit = verify(SOURCE)
    review = json.loads((ROOT/'research/one-sided-FRS-alignment-review.json').read_text())
    assert review['audit_sha256'] == digest(SOURCE/'complete-audit.json')
    assert not review['classification_scored'] and review['counts']['folds'] == 44
    return audit


def comparison(fi, b):
    old, new = b['fits']['one_sided'], b['fits']['pair_transition']
    old_bounds, pair_bounds = old['one_sided_bound_at_target'], old['pair_transition_bound_at_target']
    # Class-size weights are available in the original geometry, rather than
    # treating classes with different counts as equally weighted objects.
    counts = [r['count'] for r in b['anchor_pair_probe']['classes']]
    tightening = sum(n*(a-z) for n, a, z in zip(counts, old_bounds['raw_upper_bounds_with_inflation'],
        pair_bounds['raw_upper_bounds_with_inflation'], strict=True))/sum(counts)
    return {'fold': fi, 'objective_decrease': old['fit']['training_objective_final']-new['fit']['training_objective_final'],
        'clipped_training_bound_decrease_pp': 100*(old['fit']['training_error_upper_bound_final']-new['fit']['training_error_upper_bound_final']),
        'pair_bound_tightening_at_old_target_pp': 100*tightening,
        'new_target_outside_one_sided_cell': bool(new['pair_transition_bound_at_target']['feasible'] and not new['one_sided_bound_at_target']['feasible']),
        'old_zero_weights': int(sum(w == 0 for w in old['target'])), 'new_zero_weights': int(sum(w == 0 for w in new['target'])),
        'fit_failures': {a: int(b['fits'][a]['fit']['subproblem_failures']) for a in ARMS},
        'fit_seconds': {a: b['fits'][a]['fit_seconds'] for a in ARMS},
        'pair_enemy_rows': int(sum(len(r['enemies']) for r in b['anchor_pair_probe']['pair_transition_objects']))}


def run(name):
    threadpool_limits(1); FOLDER.mkdir(exist_ok=True); audit = prerequisite(); hashes = frozen()
    assert digest(SOURCE/(name+'.json')) == audit['artifact_hashes'][name]
    out = FOLDER/(name+'.json')
    if out.exists():
        saved = json.loads(out.read_text()); assert saved['fingerprints'] == hashes
        if saved['complete']:
            return
    source = json.loads((SOURCE/(name+'.json')).read_text()); X, y = load_dataset(name)
    ri = np.asarray(source['reference_indices']); raw, labels = X[ri], y[ri]; blocks = []
    for fi, old in enumerate(source['inner_training']):
        part = FOLDER/(name+f'.inner{fi}.partial.json')
        if part.exists():
            record = json.loads(part.read_text()); assert record['fingerprints'] == hashes
            blocks.append(record['block']); continue
        ca = np.asarray(old['reference_indices']); A, _ = preprocess_fit(raw[ca], raw[ca]); full = np.ones(A.shape[1])/np.sqrt(A.shape[1])
        fs, anchor = old['features'], np.asarray(old['anchor']); _, _, probe = build_class_cell(A, labels[ca], fs, anchor, full)
        assert probe['anchor_strictly_feasible'] == old['strict_cell_available']
        block = {'reference_indices': ca.tolist(), 'features': fs, 'anchor': anchor.tolist(),
            'anchor_pair_probe': probe, 'strict_cell_available': old['strict_cell_available'], 'fits': {}}
        if block['strict_cell_available']:
            for arm, function in (('one_sided', fit_strict_one_sided_target), ('pair_transition', fit_pair_transition_target)):
                start = time.perf_counter()
                target, fit = function(A, labels[ca], fs, anchor, full_reference=full, reference_scope='original_full')
                block['fits'][arm] = {'features': fs, 'anchor': anchor.tolist(), 'target': target.tolist(), 'fit': fit,
                    'fit_seconds': time.perf_counter()-start,
                    'one_sided_bound_at_target': monotone_evaluation(probe, target),
                    'pair_transition_bound_at_target': pair_evaluation(probe, target)}
        part.write_text(json.dumps({'fingerprints': hashes, 'block': block}, default=builtin, separators=(',', ':')), encoding='utf-8')
        blocks.append(block)
        print(name, 'training fold', fi+1, 'matched residual cell objective decrease',
            comparison(fi, block)['objective_decrease'] if block['strict_cell_available'] else None, flush=True)
    saved = {'dataset': name, 'fingerprints': hashes, 'reference_indices': ri.tolist(), 'inner_training': blocks,
        'complete': True, 'goal_achieved': False, 'classification_scored': False,
        'scope': 'matched_training_residual_envelope_same_loss_solver_budget_anchor_not_prediction'}
    out.write_text(json.dumps(saved, default=builtin, separators=(',', ':')), encoding='utf-8')


def summarize(names):
    rows, total, available = [], 0, 0
    for name in names:
        saved = json.loads((FOLDER/(name+'.json')).read_text()); blocks = saved['inner_training']
        comparisons = [comparison(fi, b) for fi, b in enumerate(blocks) if b['strict_cell_available']]
        total += len(blocks); available += len(comparisons)
        rows.append({'dataset': name, 'comparisons': comparisons,
            'mean_fold_metrics': {k: float(np.mean([r[k] for r in comparisons])) if comparisons else None for k in KEYS}})
    out = {'rows': rows, 'datasets': len(names), 'inner_training_folds': total, 'available_cells': available,
        'dataset_balanced_metrics': {k: float(np.mean([r['mean_fold_metrics'][k] for r in rows if r['comparisons']])) for k in KEYS},
        'optimizer_failures': {a: int(sum(r['fit_failures'][a] for d in rows for r in d['comparisons'])) for a in ARMS},
        'artifact_hashes': {n: digest(FOLDER/(n+'.json')) for n in names}, 'complete': True, 'independent_audit_required': True,
        'classification_scored': False, 'goal_achieved': False}
    (FOLDER/'complete.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k not in ('rows', 'artifact_hashes')}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--dataset'); args = parser.parse_args()
    if args.dataset:
        run(args.dataset)
    else:
        names = sorted(prerequisite()['artifact_hashes']); assert len(names) == 15
        with ProcessPoolExecutor(max_workers=2) as executor:
            for future in as_completed([executor.submit(run, n) for n in names]):
                future.result()
        summarize(names)
