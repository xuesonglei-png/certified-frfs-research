"""Conditional proposed-budget controls; never native-method claims."""
import argparse
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_nested_geometry_forest import (ROOT, MODELS, GRIDS, PATHS, SOURCE, NATIVE, verified_native, digest, choose)
from certified_frfs.datasets import load_dataset
from certified_frfs.geometry_only_forest_stage import GeometryStageContext
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.orthogonal_forest_logistic import orthogonal_scores
from certified_frfs.forest_penalty_selection import balanced_accuracy
from review_box_iterations import paired

ANCHOR = ROOT/'results/nested-geometry-forest'
FOLDER = ROOT/'results/geometry-forest-matched-count'


def run(name):
    threadpool_limits(1)
    anchor = json.loads((ANCHOR/(name+'.json')).read_text()); assert anchor['complete']
    hashes = dict(anchor['fingerprints'])
    for rel, value in hashes.items(): assert digest(ROOT/rel) == value
    for p in (Path(__file__), ROOT/'research/geometry-forest-matched-count-protocol.md', ANCHOR/(name+'.json')):
        hashes[p.relative_to(ROOT).as_posix()] = digest(p)
    out = FOLDER/(name+'.json'); records = []
    if out.exists():
        saved = json.loads(out.read_text()); assert saved['fingerprints'] == hashes
        if saved['complete']: return
        records = saved['splits']
    source = json.loads((SOURCE/(name+'.json')).read_text()); native = verified_native(name, source)
    X, y = load_dataset(name)
    for i, s in enumerate(anchor['splits']):
        if i < len(records): continue
        ri, vi = np.array(s['reference_indices']), np.array(s['validation_indices'])
        chosen = s['results']['proposed']; anchor_row = s['trace']['proposed'][chosen['chosen_index']]
        traces = {m: [{'model_index': mi, 'selector_index': ci, 'fraction_index': 0, 'folds': []}
                      for ci in range(len(grid)) for mi in range(len(MODELS))] for m, grid in GRIDS.items()}
        inners = []
        for j, inner in enumerate(s['inner']):
            a, b = np.array(inner['reference_indices']), np.array(inner['validation_indices'])
            context = GeometryStageContext(X[ri[a]], y[ri[a]]); _, B = preprocess_fit(X[ri[a]], X[ri[b]])
            target = len(anchor_row['folds'][j]['features']); models = []; ids = {}; scores = {}; supports = {}
            for m, grid in GRIDS.items():
                supports[m] = []
                for ci, config in enumerate(grid):
                    ranking = native['splits'][i]['inner'][j]['paths'][m][ci]['ranking']
                    assert sorted(ranking) == list(range(X.shape[1]))
                    fs = sorted(ranking[:target]); supports[m].append(fs); assert len(fs) == target
                    for mi, cfg in enumerate(MODELS):
                        fit = context.classifier(fs, **cfg); key = id(fit)
                        if key not in ids:
                            ids[key] = len(models); models.append(fit)
                            scores[key] = balanced_accuracy(orthogonal_scores(B, fit), fit['classes'], y[ri[b]])
                        traces[m][ci*len(MODELS)+mi]['folds'].append({'features': fs, 'model_id': ids[key], 'clean': scores[key]})
            inners.append({'reference_indices': a.tolist(), 'validation_indices': b.tolist(), 'target_count': target,
                'models': models, 'trees': list(context.trees.values()), 'supports': supports})
        selections = {m: choose(rows) for m, rows in traces.items()}
        context = GeometryStageContext(X[ri], y[ri]); A, B = preprocess_fit(X[ri], X[vi])
        target = len(chosen['features']); results = {}
        for m, rows in traces.items():
            row = rows[selections[m]]; config = GRIDS[m][row['selector_index']]; key = json.dumps(config, sort_keys=True)
            cache = native['splits'][i]['outer_paths'][m]
            path = cache[key] if key in cache else PATHS[m](A, y[ri], **config)
            fs = sorted(path['ranking'][:target]); assert len(fs) == target
            fit = context.classifier(fs, **MODELS[row['model_index']])
            results[m] = {'chosen_index': selections[m], 'features': fs, 'model_index': row['model_index'],
                'selector_index': row['selector_index'], 'config': config, 'path': path, 'cached_path': key in cache, 'fit': fit}
        rng = np.random.default_rng(144+i)
        queries = [B]+[np.clip(B+rng.uniform(-.01, .01, B.shape), 0, 1) for _ in range(3)]
        for r in results.values():
            values = [balanced_accuracy(orthogonal_scores(Q, r['fit']), r['fit']['classes'], y[vi]) for Q in queries]
            r['metrics'] = {'clean': values[0], 'noise': float(np.mean(values[1:]))}
        records.append({'reference_indices': ri.tolist(), 'validation_indices': vi.tolist(), 'target_count': target,
            'anchor_chosen_index': chosen['chosen_index'], 'inner': inners, 'trace': traces, 'results': results,
            'trees': list(context.trees.values()), 'query_seed': 144+i})
        out.write_text(json.dumps({'dataset': name, 'd': X.shape[1], 'splits': records, 'fingerprints': hashes,
            'complete': len(records) == len(anchor['splits']), 'scope': 'conditional_exact_count_adaptations_not_native_methods'}, separators=(',', ':')), encoding='utf-8')
        print(name, i, 'matched complete', {m: r['metrics']['clean'] for m, r in results.items()}, flush=True)


def summarize(names):
    rows = []
    for name in names:
        saved = json.loads((FOLDER/(name+'.json')).read_text()); anchor = json.loads((ANCHOR/(name+'.json')).read_text())
        assert saved['complete'] and len(saved['splits']) == len(anchor['splits'])
        methods = {m: {metric: float(np.mean([s['results'][m]['metrics'][metric] for s in saved['splits']])) for metric in ('clean', 'noise')} for m in GRIDS}
        methods['proposed'] = {metric: float(np.mean([s['results']['proposed']['metrics'][metric] for s in anchor['splits']])) for metric in ('clean', 'noise')}
        for s, a in zip(saved['splits'], anchor['splits']):
            assert s['target_count'] == len(a['results']['proposed']['features'])
            for m in GRIDS: assert len(s['results'][m]['features']) == s['target_count']
        rows.append({'dataset': name, 'methods': methods})
    summary = {'rows': rows, 'contrasts': {metric: {m: paired([r['methods']['proposed'][metric]-r['methods'][m][metric] for r in rows]) for m in GRIDS} for metric in ('clean', 'noise')},
        'outer_splits': sum(len(json.loads((FOLDER/(name+'.json')).read_text())['splits']) for name in names),
        'scope': 'development_conditional_exact_count_adaptations_not_native_methods_budget_chosen_by_proposed_inner_pipeline'}
    (FOLDER/'complete.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print({k: v for k, v in summary.items() if k != 'rows'}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--only'); args = parser.parse_args()
    FOLDER.mkdir(exist_ok=True)
    names = [r['dataset'] for r in json.loads((SOURCE/'complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run, n) for n in ([args.only] if args.only else names)]): f.result()
    if not args.only: summarize(names)
