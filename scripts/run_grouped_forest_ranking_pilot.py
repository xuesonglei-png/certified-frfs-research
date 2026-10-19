"""All128 group-shrunk selector pilot with a common final classifier."""
import argparse
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_nested_geometry_forest import ROOT, SOURCE, digest
from certified_frfs.grouped_geometry_forest_stage import GroupedGeometryStageContext
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.orthogonal_forest_logistic import orthogonal_scores
from certified_frfs.forest_penalty_selection import balanced_accuracy
from review_box_iterations import paired

ANCHOR = ROOT/'results/nested-geometry-forest'
FOLDER = ROOT/'results/grouped-forest-ranking-pilot'
METHODS = ('old', 'grouped', 'matched', 'full')


def run(name):
    threadpool_limits(1); anchor = json.loads((ANCHOR/(name+'.json')).read_text()); assert anchor['complete']
    hashes = dict(anchor['fingerprints'])
    for rel, value in hashes.items(): assert digest(ROOT/rel) == value
    for p in (Path(__file__), ROOT/'research/grouped-forest-ranking-pilot-protocol.md', ANCHOR/(name+'.json'),
            ROOT/'src/certified_frfs/grouped_geometry_forest_stage.py', ROOT/'src/certified_frfs/grouped_orthogonal_forest.py'):
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
            A, B = preprocess_fit(X[ri[a]], X[ri[b]]); labels = y[ri[a]]
            old_stage = inner['stages'][7]; assert old_stage['C'] == old_stage['pair_penalty'] == 1.
            old = inner['models'][old_stage['policies']['0.8']['model_id']]
            context = GroupedGeometryStageContext(X[ri[a]], labels)
            context.geometry = {tuple(g['features']): g for g in inner['geometry']}
            context.trees = {tuple(t['features']): t for t in inner['trees']}
            stage = context.fit_stage(C=1., pair_penalty=1., relative_strength=.1)
            grouped = stage['policies']['0.8']['fit']
            matched = next(v for v in reversed(stage['path']['trace']) if len(v['features']) >= len(old['features']))
            fits = {'old': old, 'grouped': grouped, 'matched': context.classifier(matched['features'], C=1., pair_penalty=1.),
                'full': inner['models'][old_stage['policies']['1.0']['model_id']]}
            rng = np.random.default_rng(144+i); queries = [B]+[np.clip(B+rng.uniform(-.01, .01, B.shape), 0, 1) for _ in range(3)]
            metrics = {}
            for m, fit in fits.items():
                values = [balanced_accuracy(orthogonal_scores(Q, fit), fit['classes'], y[ri[b]]) for Q in queries]
                metrics[m] = {'clean': values[0], 'noise': float(np.mean(values[1:])), 'removed': 1-len(fit['features'])/X.shape[1]}
            records.append({'outer_index': i, 'inner_index': j, 'outer_reference_indices': ri.tolist(),
                'reference_indices': a.tolist(), 'validation_indices': b.tolist(), 'query_seed': 144+i,
                'stage': stage, 'fits': fits, 'metrics': metrics, 'matched_count_exact': len(matched['features']) == len(old['features'])})
            out.write_text(json.dumps({'dataset': name, 'd': X.shape[1], 'fingerprints': hashes, 'records': records,
                'complete': len(records) == sum(len(s['inner']) for s in anchor['splits']), 'scope': 'fixed_config_all128_development_group_shrunk_selector'}, separators=(',', ':')), encoding='utf-8')
            print(name, i, j, 'group pilot', {m: r['clean'] for m, r in metrics.items()}, flush=True)


def summarize(names):
    rows = []; counts = {'records': 0, 'selector_fallbacks': 0, 'final_fit_caps': 0, 'matched_count_mismatches': 0}
    for name in names:
        saved = json.loads((FOLDER/(name+'.json')).read_text()); assert saved['complete']
        counts['records'] += len(saved['records'])
        counts['selector_fallbacks'] += sum(r['stage']['group_selector_fallback'] for r in saved['records'])
        counts['final_fit_caps'] += sum(not f['optimizer_converged'] for r in saved['records'] for f in r['fits'].values())
        counts['matched_count_mismatches'] += sum(not r['matched_count_exact'] for r in saved['records'])
        rows.append({'dataset': name, 'methods': {m: {metric: float(np.mean([r['metrics'][m][metric] for r in saved['records']])) for metric in ('clean', 'noise', 'removed')} for m in METHODS}})
    assert counts['records'] == 128 and len(rows) == 15
    result = {'rows': rows, 'counts': counts,
        'means': {m: {metric: float(np.mean([r['methods'][m][metric] for r in rows])) for metric in ('clean', 'noise', 'removed')} for m in METHODS},
        'contrasts': {metric: {m: paired([r['methods'][m][metric]-r['methods']['old'][metric] for r in rows]) for m in METHODS[1:]} for metric in ('clean', 'noise')},
        'scope': 'all15_all128_fixed_config_exposed_development_no_nested_or_SOTA_confirmation'}
    result['gate_passed'] = result['contrasts']['clean']['grouped']['mean_pp'] >= .5 and result['means']['grouped']['removed'] >= result['means']['old']['removed'] and counts['selector_fallbacks'] == counts['final_fit_caps'] == 0
    (FOLDER/'complete.json').write_text(json.dumps(result, indent=2), encoding='utf-8'); print({k:v for k,v in result.items() if k != 'rows'}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--only'); args = parser.parse_args()
    FOLDER.mkdir(exist_ok=True); names = [r['dataset'] for r in json.loads((SOURCE/'complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run, n) for n in ([args.only] if args.only else names)]): f.result()
    if not args.only: summarize(names)
