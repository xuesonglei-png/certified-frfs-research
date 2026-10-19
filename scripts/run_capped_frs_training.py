"""ALL15/44 capped objective and warm initialization; no prediction scoring."""
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
from fit_capped_frs_warm import fit_capped_frs_warm_target

SOURCE = ROOT/'results/frs-one-sided-training'
FOLDER = ROOT/'results/capped-frs-training'
FREEZE = ROOT/'research/capped-frs-training-source-freeze.json'


def builtin(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def frozen():
    hashes = json.loads(FREEZE.read_text())['fingerprints']
    for rel, sha in hashes.items():
        assert digest(ROOT/rel) == sha, rel
    return hashes


def prerequisite(name):
    audit_path = SOURCE/'complete-audit.json'
    if audit_path.exists():
        assert verify(SOURCE)['artifact_hashes'][name] == digest(SOURCE/(name+'.json'))
    else:
        audit = json.loads((SOURCE/(name+'.smoke-audit.json')).read_text())
        assert audit['passed'] and audit['sha256'] == digest(SOURCE/(name+'.json'))


def run(name):
    threadpool_limits(1); FOLDER.mkdir(exist_ok=True); prerequisite(name)
    hashes = {**frozen(), (SOURCE/(name+'.json')).relative_to(ROOT).as_posix(): digest(SOURCE/(name+'.json'))}
    out = FOLDER/(name+'.json')
    if out.exists():
        saved = json.loads(out.read_text()); assert saved['fingerprints'] == hashes
        if saved['complete']:
            return
    source = json.loads((SOURCE/(name+'.json')).read_text()); X, y = load_dataset(name)
    reference = np.asarray(source['reference_indices']); raw, labels = X[reference], y[reference]; blocks = []
    for fi, old in enumerate(source['inner_training']):
        part = FOLDER/(name+f'.inner{fi}.partial.json')
        if part.exists():
            stored = json.loads(part.read_text()); assert stored['fingerprints'] == hashes; blocks.append(stored['block']); continue
        ca = np.asarray(old['reference_indices']); A, _ = preprocess_fit(raw[ca], raw[ca]); full = np.ones(A.shape[1])/np.sqrt(A.shape[1])
        fs, anchor = old['features'], np.asarray(old['anchor'])
        block = {'reference_indices': ca.tolist(), 'features': fs, 'anchor': anchor.tolist(),
            'strict_cell_available': old['strict_cell_available'], 'fits': {}, 'training_selected_start': None}
        if old['strict_cell_available']:
            warm = np.asarray(old['fits']['one_sided']['target'])
            for family, initialization, scope in [('anchor', None, 'anchor'),
                    ('warm', warm, 'training_one_sided_unbounded_endpoint')]:
                start = time.perf_counter()
                target, fit = fit_capped_frs_warm_target(A, labels[ca], fs, anchor, full_reference=full,
                    reference_scope='original_full', initialization=initialization, initialization_scope=scope)
                block['fits'][family] = {'features': fs, 'anchor': anchor.tolist(), 'target': target.tolist(),
                    'fit': fit, 'fit_seconds': time.perf_counter()-start}
            block['training_selected_start'] = min(('anchor', 'warm'),
                key=lambda family: (block['fits'][family]['fit']['training_objective_final'], 0 if family == 'anchor' else 1))
        part.write_text(json.dumps({'fingerprints': hashes, 'block': block}, default=builtin, separators=(',', ':')), encoding='utf-8')
        blocks.append(block)
        print(name, 'training fold', fi+1, 'selected start', block['training_selected_start'], 'failures',
            sum(r['fit']['subproblem_failures'] for r in block['fits'].values()), flush=True)
    saved = {'dataset': name, 'fingerprints': hashes, 'reference_indices': reference.tolist(), 'inner_training': blocks,
        'complete': True, 'goal_achieved': False, 'classification_scored': False,
        'scope': 'training_only_bounded_error_objective_and_fixed_two_initializations'}
    out.write_text(json.dumps(saved, default=builtin, separators=(',', ':')), encoding='utf-8')


def dataset_comparison(saved, source):
    comparisons = []
    for fi, (b, old) in enumerate(zip(saved['inner_training'], source['inner_training'], strict=True)):
        if not b['strict_cell_available']:
            continue
        new = b['fits'][b['training_selected_start']]['fit']; prior = old['fits']['one_sided']
        anchor = np.asarray(b['anchor']); target = np.asarray(prior['target']); reg = .01*float(abs(target-anchor).sum())/new['mass']
        prior_error = prior['fit']['training_error_upper_bound_final']; error = new['training_error_upper_bound_final']
        comparisons.append({'fold': fi, 'selected_start': b['training_selected_start'],
            'prior_regularized_clipped_objective': prior_error+reg, 'new_regularized_clipped_objective': new['training_objective_final'],
            'clipped_objective_decrease_vs_previous': prior_error+reg-new['training_objective_final'],
            'prior_clipped_training_bound': prior_error, 'new_clipped_training_bound': error,
            'clipped_training_bound_decrease_pp': 100*(prior_error-error),
            'anchor_to_start_objective_decrease': new['training_objective_initial']-new['optimization_start_objective'],
            'start_to_final_objective_decrease': new['optimization_start_objective']-new['training_objective_final'],
            'new_exact_zero_weights': int(sum(w == 0 for w in new['weights'])),
            'failures': sum(r['fit']['subproblem_failures'] for r in b['fits'].values())})
    return {'dataset': saved['dataset'], 'comparisons': comparisons,
        'mean_clipped_training_bound_decrease_pp': float(np.mean([r['clipped_training_bound_decrease_pp'] for r in comparisons])) if comparisons else None,
        'mean_clipped_objective_decrease': float(np.mean([r['clipped_objective_decrease_vs_previous'] for r in comparisons])) if comparisons else None}


def summarize(names):
    rows, folds, available, failures = [], 0, 0, 0
    for name in names:
        saved = json.loads((FOLDER/(name+'.json')).read_text()); source = json.loads((SOURCE/(name+'.json')).read_text())
        row = dataset_comparison(saved, source); rows.append(row)
        folds += len(saved['inner_training']); available += len(row['comparisons']); failures += sum(r['failures'] for r in row['comparisons'])
    out = {'rows': rows, 'datasets': len(names), 'inner_training_folds': folds, 'available_cells': available, 'optimizer_failures': failures,
        'mean_dataset_clipped_training_bound_decrease_pp': float(np.mean([r['mean_clipped_training_bound_decrease_pp'] for r in rows if r['comparisons']])),
        'mean_dataset_clipped_objective_decrease': float(np.mean([r['mean_clipped_objective_decrease'] for r in rows if r['comparisons']])),
        'artifact_hashes': {n: digest(FOLDER/(n+'.json')) for n in names}, 'complete': True,
        'goal_achieved': False, 'classification_scored': False, 'independent_audit_required': True}
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
