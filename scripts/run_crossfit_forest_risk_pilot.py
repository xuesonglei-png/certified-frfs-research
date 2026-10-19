"""All128 fixed-config crossfit-risk path pilot; exposed development only."""
import argparse
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_nested_geometry_forest import ROOT, SOURCE, digest
from certified_frfs.datasets import load_dataset
from certified_frfs.crossfit_geometry_forest_stage import CrossfitGeometryStageContext
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.orthogonal_forest_logistic import orthogonal_scores
from certified_frfs.forest_penalty_selection import balanced_accuracy
from review_box_iterations import paired

ANCHOR = ROOT/'results/nested-geometry-forest'
FOLDER = ROOT/'results/crossfit-forest-risk-pilot'


def run(name):
    threadpool_limits(1)
    anchor = json.loads((ANCHOR/(name+'.json')).read_text()); assert anchor['complete']
    hashes = dict(anchor['fingerprints'])
    for rel, value in hashes.items(): assert digest(ROOT/rel) == value
    for p in (Path(__file__), ROOT/'research/crossfit-forest-risk-pilot-protocol.md', ANCHOR/(name+'.json'),
            ROOT/'src/certified_frfs/crossfit_geometry_forest_stage.py', ROOT/'src/certified_frfs/crossfit_forest_risk.py',
            ROOT/'src/certified_frfs/nested_teacher_regularized.py'):
        hashes[p.relative_to(ROOT).as_posix()] = digest(p)
    out = FOLDER/(name+'.json'); records = []
    if out.exists():
        saved = json.loads(out.read_text()); assert saved['fingerprints'] == hashes
        if saved['complete']: return
        records = saved['records']
    X, y = load_dataset(name); n = 0
    for i, s in enumerate(anchor['splits']):
        ri = np.array(s['reference_indices'])
        for j, inner in enumerate(s['inner']):
            n += 1
            if n <= len(records): continue
            a, b = np.array(inner['reference_indices']), np.array(inner['validation_indices'])
            raw, labels = X[ri[a]], y[ri[a]]; A, B = preprocess_fit(raw, X[ri[b]])
            stage = inner['stages'][7]; assert stage['C'] == 1. and stage['pair_penalty'] == 1.
            old_policy = stage['policies']['0.8']; old_fit = inner['models'][old_policy['model_id']]
            old_fs = old_fit['features']; context = CrossfitGeometryStageContext(raw, labels)
            # Identical training table/support certificate cache, never scores.
            context.geometry = {tuple(g['features']): g for g in inner['geometry']}
            context.trees = {tuple(t['features']): t for t in inner['trees']}
            fresh = context.fit_stage(C=1., pair_penalty=1.)
            new_policy = fresh['policies']['0.8']; new_fit = new_policy['fit']
            matched = next(v for v in reversed(fresh['path']['trace']) if len(v['features']) >= len(old_fs))
            matched_fit = context.classifier(matched['features'], C=1., pair_penalty=1.)
            full_fit = inner['models'][stage['policies']['1.0']['model_id']]
            fits = {'old': old_fit, 'crossfit': new_fit, 'matched': matched_fit, 'full': full_fit}
            rng = np.random.default_rng(144+i)
            queries = [B]+[np.clip(B+rng.uniform(-.01, .01, B.shape), 0, 1) for _ in range(3)]
            metrics = {}
            for m, fit in fits.items():
                values = [balanced_accuracy(orthogonal_scores(Q, fit), fit['classes'], y[ri[b]]) for Q in queries]
                metrics[m] = {'clean': values[0], 'noise': float(np.mean(values[1:])), 'removed': 1-len(fit['features'])/X.shape[1]}
            records.append({'outer_index': i, 'inner_index': j, 'outer_reference_indices': ri.tolist(),
                'reference_indices': a.tolist(), 'validation_indices': b.tolist(), 'query_seed': 144+i,
                'stage': fresh, 'fits': fits, 'matched_count_exact': len(matched['features']) == len(old_fs),
                'metrics': metrics})
            out.write_text(json.dumps({'dataset': name, 'd': X.shape[1], 'fingerprints': hashes, 'records': records,
                'complete': len(records) == sum(len(s['inner']) for s in anchor['splits']),
                'scope': 'fixed_config_all128_exposed_development_crossfit_risk_ranking'}, separators=(',', ':')), encoding='utf-8')
            print(name, i, j, 'risk pilot', {m: r['clean'] for m, r in metrics.items()}, flush=True)


def summarize(names):
    rows = []; records = 0; mismatches = 0; caps = 0; fold_caps = 0
    for name in names:
        saved = json.loads((FOLDER/(name+'.json')).read_text()); assert saved['complete']
        records += len(saved['records']); mismatches += sum(not r['matched_count_exact'] for r in saved['records'])
        caps += sum(not fit['optimizer_converged'] for r in saved['records'] for fit in r['fits'].values())
        fold_caps += sum(not f['fit']['optimizer_converged'] for r in saved['records'] for f in r['stage']['path']['crossfit_splits'])
        rows.append({'dataset': name, 'methods': {m: {metric: float(np.mean([r['metrics'][m][metric] for r in saved['records']])) for metric in ('clean', 'noise', 'removed')} for m in ('old', 'crossfit', 'matched', 'full')}})
    assert records == 128 and len(rows) == 15
    result = {'rows': rows, 'records': records, 'matched_count_mismatches': mismatches, 'final_fit_caps': caps, 'fold_teacher_caps': fold_caps,
        'contrasts': {metric: {m: paired([r['methods'][m][metric]-r['methods']['old'][metric] for r in rows]) for m in ('crossfit', 'matched', 'full')} for metric in ('clean', 'noise')},
        'means': {m: {metric: float(np.mean([r['methods'][m][metric] for r in rows])) for metric in ('clean', 'noise', 'removed')} for m in ('old', 'crossfit', 'matched', 'full')},
        'scope': 'all128_fixed_config_development_not_nested_confirmation_or_risk_bound'}
    result['gate_passed'] = result['contrasts']['clean']['crossfit']['mean_pp'] >= .5 and result['means']['crossfit']['removed'] >= result['means']['old']['removed'] and caps == 0 and fold_caps == 0
    (FOLDER/'complete.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print({k: v for k, v in result.items() if k != 'rows'}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--only'); args = parser.parse_args()
    FOLDER.mkdir(exist_ok=True); names = [r['dataset'] for r in json.loads((SOURCE/'complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run, n) for n in ([args.only] if args.only else names)]): f.result()
    if not args.only: summarize(names)
