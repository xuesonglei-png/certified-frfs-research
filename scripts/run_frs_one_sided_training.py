"""ALL15/44 same-anchor envelope ablation; no validation or test scoring."""
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
from fit_frs_cell_hinge import fit_frs_cell_target
from fit_frs_one_sided_hinge import fit_one_sided_cell_target
from frs_one_sided_cell import build_class_cell, evaluate_class_cell
from frs_class_cell import evaluate_class_cell as symmetric_evaluation

SOURCE = ROOT/'results/atomic-pruning-classifier-v2'
FOLDER = ROOT/'results/frs-one-sided-training'
FREEZE = ROOT/'research/frs-one-sided-training-source-freeze.json'


def builtin(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def frozen():
    hashes = json.loads(FREEZE.read_text())['fingerprints']
    for rel, sha in hashes.items():
        assert digest(ROOT/rel) == sha, rel
    return hashes


def run(name):
    threadpool_limits(1); FOLDER.mkdir(exist_ok=True)
    assert verify(SOURCE)['artifact_hashes'][name] == digest(SOURCE/(name+'.json'))
    hashes = {**frozen(), (SOURCE/(name+'.json')).relative_to(ROOT).as_posix(): digest(SOURCE/(name+'.json'))}
    output = FOLDER/(name+'.json')
    if output.exists():
        saved = json.loads(output.read_text()); assert saved['fingerprints'] == hashes
        if saved['complete']:
            return
    source = json.loads((SOURCE/(name+'.json')).read_text()); X, y = load_dataset(name)
    reference = np.asarray(source['reference_indices']); raw, labels = X[reference], y[reference]; blocks = []
    for fi, old in enumerate(source['inner']):
        part = FOLDER/(name+f'.inner{fi}.partial.json')
        if part.exists():
            saved = json.loads(part.read_text()); assert saved['fingerprints'] == hashes; blocks.append(saved['block']); continue
        ca = np.asarray(old['reference_indices']); A, _ = preprocess_fit(raw[ca], raw[ca]); training_y = labels[ca]
        base = old['options']['proposed'][source['selection']['selected']['proposed']]['metric']
        fs, anchor = base['features'], np.asarray(base['weights']); full = np.ones(A.shape[1])/np.sqrt(A.shape[1])
        _, _, probe = build_class_cell(A, training_y, fs, anchor, full)
        block = {'reference_indices': ca.tolist(), 'features': fs, 'anchor': anchor.tolist(),
            'strict_cell_available': bool(probe['anchor_strictly_feasible']), 'anchor_probe': probe, 'fits': {}}
        if probe['anchor_strictly_feasible']:
            for family, fit_function in [('symmetric', fit_frs_cell_target), ('one_sided', fit_one_sided_cell_target)]:
                start = time.perf_counter()
                target, fit = fit_function(A, training_y, fs, anchor, full_reference=full, reference_scope='original_full')
                block['fits'][family] = {'features': fs, 'anchor': anchor.tolist(), 'target': target.tolist(), 'fit': fit,
                    'fit_seconds': time.perf_counter()-start,
                    'symmetric_bound_at_target': symmetric_evaluation(fit['class_cell'], target),
                    'one_sided_bound_at_target': evaluate_class_cell(fit['class_cell'], target)}
        part.write_text(json.dumps({'fingerprints': hashes, 'block': block}, default=builtin, separators=(',', ':')), encoding='utf-8')
        blocks.append(block)
        print(name, 'training fold', fi+1, 'cell available', block['strict_cell_available'], flush=True)
    saved = {'dataset': name, 'fingerprints': hashes, 'reference_indices': reference.tolist(), 'inner_training': blocks,
        'complete': True, 'goal_achieved': False, 'classification_scored': False,
        'scope': 'training_only_fixed_support_same_anchor_symmetric_vs_monotone_certificate'}
    output.write_text(json.dumps(saved, default=builtin, separators=(',', ':')), encoding='utf-8')


def summarize(names):
    rows = []; total_folds, available, failures = 0, 0, 0
    for name in names:
        saved = json.loads((FOLDER/(name+'.json')).read_text()); comparisons = []
        for fi, b in enumerate(saved['inner_training']):
            total_folds += 1; available += b['strict_cell_available']
            if not b['strict_cell_available']:
                continue
            old, new = b['fits']['symmetric'], b['fits']['one_sided']
            failures += old['fit']['subproblem_failures']+new['fit']['subproblem_failures']
            comparisons.append({'fold': fi, 'old_objective': old['fit']['training_objective_final'],
                'new_objective': new['fit']['training_objective_final'],
                'objective_decrease_vs_symmetric': old['fit']['training_objective_final']-new['fit']['training_objective_final'],
                'new_target_outside_symmetric_cell': bool(new['one_sided_bound_at_target']['feasible'] and not new['symmetric_bound_at_target']['feasible']),
                'failures': old['fit']['subproblem_failures']+new['fit']['subproblem_failures']})
        rows.append({'dataset': name, 'comparisons': comparisons,
            'mean_training_objective_decrease_vs_symmetric': float(np.mean([r['objective_decrease_vs_symmetric'] for r in comparisons])) if comparisons else None})
    out = {'rows': rows, 'datasets': len(names), 'inner_training_folds': total_folds, 'available_cells': int(available),
        'optimizer_failures': failures, 'artifact_hashes': {n: digest(FOLDER/(n+'.json')) for n in names},
        'complete': True, 'goal_achieved': False, 'classification_scored': False, 'independent_audit_required': True}
    (FOLDER/'complete.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k not in ('rows', 'artifact_hashes')}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--dataset'); args = parser.parse_args()
    if args.dataset:
        run(args.dataset)
    else:
        names = sorted(verify(SOURCE)['artifact_hashes']); assert len(names) == 15
        with ProcessPoolExecutor(max_workers=2) as executor:
            for f in as_completed([executor.submit(run, n) for n in names]):
                f.result()
        summarize(names)
