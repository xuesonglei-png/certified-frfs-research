"""ALL15 matched fixed-three/latent-five training; requires complete audits."""
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
from latent_frs_training_ablation import fit_block

SOURCE = ROOT/'results/capped-frs-training'
FOLDER = ROOT/'results/latent-frs-training'
FREEZE = ROOT/'research/latent-frs-training-source-freeze.json'
REVIEW = ROOT/'research/capped-frs-training-review.json'
ARMS = ('fixed_three', 'latent_five')
KEYS = ('selected_training_bound_decrease_vs_fixed_three_pp',
        'same_weight_tightening_at_fixed_three_endpoint_pp',
        'latent_objective_improvement_beyond_fixed_three_endpoint',
        'regularized_training_objective_decrease_vs_fixed_three')


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
    audit = verify(SOURCE); review = json.loads(REVIEW.read_text())
    assert review['audit_sha256'] == digest(SOURCE/'complete-audit.json')
    assert not review['classification_scored'] and not review['goal_achieved']
    assert len(audit['artifact_hashes']) == 15
    return audit


def dataset_row(name, blocks):
    folds = [{'fold': i, **b['evaluation']} for i, b in enumerate(blocks) if b['strict_cell_available']]
    return {'dataset': name, 'folds': folds,
        'mean_fold_metrics': {k: float(np.mean([r[k] for r in folds])) if folds else None for k in KEYS},
        'optimizer_failures': {a: int(sum(r['optimizer_failures'][a] for r in folds)) for a in ARMS}}


def run(name):
    threadpool_limits(1); FOLDER.mkdir(exist_ok=True)
    audit = prerequisite(); hashes = frozen()
    assert name in audit['artifact_hashes'] and digest(SOURCE/(name+'.json')) == audit['artifact_hashes'][name]
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
            stored = json.loads(part.read_text()); assert stored['fingerprints'] == hashes
            blocks.append(stored['block']); continue
        ca = np.asarray(old['reference_indices']); A, _ = preprocess_fit(raw[ca], raw[ca])
        timer = time.perf_counter()
        if old['strict_cell_available']:
            prior_fit = old['fits'][old['training_selected_start']]
            block = fit_block(A, labels[ca], old['features'], old['anchor'],
                np.ones(A.shape[1])/np.sqrt(A.shape[1]), prior_fit['target'])
            block['prior_selected_start'] = old['training_selected_start']
            block['prior_selected_fit_optimizer_success'] = prior_fit['fit']['optimizer_success']
            block['prior_fit_failures'] = sum(r['fit']['subproblem_failures'] for r in old['fits'].values())
        else:
            block = {'features': old['features'], 'anchor': old['anchor'], 'arms': {}, 'evaluation': None,
                'capped_endpoint': None, 'prior_selected_start': None, 'prior_fit_failures': 0,
                'prior_selected_fit_optimizer_success': None}
        block.update({'reference_indices': ca.tolist(), 'strict_cell_available': old['strict_cell_available'],
                      'fit_seconds': time.perf_counter()-timer})
        part.write_text(json.dumps({'fingerprints': hashes, 'block': block}, default=builtin, separators=(',', ':')), encoding='utf-8')
        blocks.append(block)
        print(name, 'training fold', fi+1, 'latent-minus-fixed bound improvement pp',
            None if block['evaluation'] is None else block['evaluation'][KEYS[0]], flush=True)
    saved = {'dataset': name, 'fingerprints': hashes, 'reference_indices': ri.tolist(), 'inner_training': blocks,
        'complete': True, 'goal_achieved': False, 'classification_scored': False,
        'scope': 'matched_strict_solver_two_start_training_witness_ablation_not_test_advantage'}
    out.write_text(json.dumps(saved, default=builtin, separators=(',', ':')), encoding='utf-8')


def summarize(names):
    rows, total, available = [], 0, 0
    for name in names:
        saved = json.loads((FOLDER/(name+'.json')).read_text()); b = saved['inner_training']
        rows.append(dataset_row(name, b)); total += len(b); available += sum(r['strict_cell_available'] for r in b)
    summary = {'rows': rows, 'datasets': len(names), 'inner_training_folds': total, 'available_cells': available,
        'dataset_balanced_metrics': {k: float(np.mean([r['mean_fold_metrics'][k] for r in rows if r['folds']])) for k in KEYS},
        'optimizer_failures': {a: int(sum(r['optimizer_failures'][a] for r in rows)) for a in ARMS},
        'artifact_hashes': {n: digest(FOLDER/(n+'.json')) for n in names},
        'complete': True, 'independent_audit_required': True, 'classification_scored': False, 'goal_achieved': False}
    (FOLDER/'complete.json').write_text(json.dumps(summary, indent=2, default=builtin), encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('rows', 'artifact_hashes')}, indent=2), flush=True)


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
