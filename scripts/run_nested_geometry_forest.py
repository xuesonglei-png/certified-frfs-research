"""Fresh nested certified supports and identical downstream classifier searches."""
import argparse
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from review_box_iterations import ROOT, paired
from compare_trajectory_recent import GRIDS
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.geometry_only_forest_stage import GeometryStageContext, FRACTIONS
from certified_frfs.orthogonal_forest_logistic import orthogonal_scores
from certified_frfs.forest_penalty_selection import balanced_accuracy
from certified_frfs.recent_budget import fgs_budget_path, sfrip_budget_path

SOURCE = ROOT/'results/nested-crossfit-classification-rounded'
NATIVE = ROOT/'results/common-fold-recent'
FOLDER = ROOT/'results/nested-geometry-forest'
CS = (.1, 1., 10.)
PAIRS = (None, 10., 3., 1.)
MODELS = [{'C': C, 'pair_penalty': g} for C in CS for g in PAIRS]
PATHS = {'fgs2023': fgs_budget_path, 'sfrip2025': sfrip_budget_path}
METHODS = ('proposed', 'full', 'fgs2023', 'sfrip2025')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprints(name):
    files = [Path(__file__), ROOT/'research/nested-geometry-forest-protocol.md', SOURCE/(name+'.json'),
        NATIVE/(name+'.json'), ROOT/'scripts/compare_trajectory_recent.py', ROOT/'data/processed'/(name+'.npz')]
    files += [ROOT/'src/certified_frfs'/f for f in ('geometry_only_forest_stage.py', 'orthogonal_forest_logistic.py',
        'fast_forest_projection.py', 'fuzzy_interaction_tree.py', 'forest_fuzzy_classifier.py', 'triangular_classifier.py',
        'validated_pool.py', 'feasible_transport.py', 'redistributed_loss.py', 'capped_sparse.py',
        'majority_metric.py', 'recent_budget.py', 'recent.py', 'graph_recent.py', 'datasets.py')]
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in files}


def verified_native(name, source):
    saved = json.loads((NATIVE/(name+'.json')).read_text())
    for rel, value in saved['fingerprints'].items():
        assert digest(ROOT/rel.replace('\\', '/')) == value, rel
    assert len(saved['splits']) == len(source['splits'])
    for s, original in zip(saved['splits'], source['splits']):
        for key in ('reference_indices', 'validation_indices'): assert s[key] == original[key]
        assert len(s['inner']) == len(original['inner'])
        for a, b in zip(s['inner'], original['inner']):
            for key in ('reference_indices', 'validation_indices'): assert a[key] == b[key]
        for method, grid in GRIDS.items():
            for j, config in enumerate(grid): assert s['trace'][method]['native'][3*j]['config'] == config
    return saved


class Registry:
    def __init__(self, context, B=None, truth=None):
        self.context = context; self.models = []; self.ids = {}; self.scores = {}
        self.B = B; self.truth = truth

    def capture(self, fit):
        key = id(fit)
        if key not in self.ids:
            self.ids[key] = len(self.models); self.models.append(fit)
        return self.ids[key]

    def score(self, fit):
        mid = self.capture(fit)
        if mid not in self.scores:
            self.scores[mid] = balanced_accuracy(orthogonal_scores(self.B, fit), fit['classes'], self.truth)
        return mid, self.scores[mid]

    def stage(self, stage):
        # Store references rather than copying large geometry certificates into
        # every parameter's path and every score row.
        geometries = list(self.context.geometry.values()); gids = {tuple(v['features']): i for i, v in enumerate(geometries)}
        def candidate(r): return {'geometry_id': gids[tuple(r['features'])], 'projection_loss': r['projection_loss']}
        return {'C': stage['C'], 'pair_penalty': stage['pair_penalty'],
            'original_majority_protected_counts': stage['original_majority_protected_counts'],
            'policies': {f: {'geometry_id': gids[tuple(r['candidate']['features'])], 'model_id': self.capture(r['fit'])}
                         for f, r in stage['policies'].items()},
            'path': {'trace': [candidate(r) for r in stage['path']['trace']],
                'attempts': [{**candidate(r), 'removed': r['removed'], 'accepted': r['accepted']} for r in stage['path']['attempts']],
                'rankings': stage['path']['rankings'], 'center': stage['path']['center'], 'tie_decimals': 12},
            'teacher_guards_enforced': False, 'scope': stage['scope']}

    def export(self):
        return {'models': self.models, 'geometry': list(self.context.geometry.values()),
            'trees': list(self.context.trees.values())}


def traces():
    result = {'proposed': [{'model_index': mi, 'fraction': f, 'selector_index': 0, 'fraction_index': fi, 'folds': []}
                           for mi in range(len(MODELS)) for fi, f in enumerate(FRACTIONS)],
              'full': [{'model_index': mi, 'selector_index': 0, 'fraction_index': 0, 'folds': []} for mi in range(len(MODELS))]}
    for method, grid in GRIDS.items():
        result[method] = [{'model_index': mi, 'selector_index': ci, 'fraction_index': 0, 'folds': []}
                          for ci in range(len(grid)) for mi in range(len(MODELS))]
    return result


def choose(rows):
    for r in rows:
        r['mean_clean'] = float(np.mean([v['clean'] for v in r['folds']]))
        r['mean_size'] = float(np.mean([len(v['features']) for v in r['folds']]))
    return max(range(len(rows)), key=lambda j: (round(rows[j]['mean_clean'], 12), -rows[j]['mean_size'],
        -rows[j]['model_index'], -rows[j]['selector_index'], -rows[j]['fraction_index']))


def run(name):
    threadpool_limits(1)
    hashes = fingerprints(name); out = FOLDER/(name+'.json')
    if out.exists():
        saved = json.loads(out.read_text()); assert saved['fingerprints'] == hashes
        if saved.get('complete'): return
        records = saved['splits']
    else: records = []
    source = json.loads((SOURCE/(name+'.json')).read_text()); native = verified_native(name, source)
    X, y = load_dataset(name)
    for i, s in enumerate(source['splits']):
        if i < len(records):
            for key in ('reference_indices', 'validation_indices'): assert records[i][key] == s[key]
            continue
        ri, vi = np.array(s['reference_indices']), np.array(s['validation_indices'])
        raw, labels = X[ri], y[ri]; selections = traces(); inner_records = []
        for j, fold in enumerate(s['inner']):
            a, b = np.array(fold['reference_indices']), np.array(fold['validation_indices'])
            assert set(a).isdisjoint(b) and len(set(a)|set(b)) == len(raw)
            context = GeometryStageContext(raw[a], labels[a]); A, B = preprocess_fit(raw[a], raw[b])
            np.testing.assert_array_equal(A, context.A)
            registry = Registry(context, B, labels[b]); stages = []
            for mi, cfg in enumerate(MODELS):
                stage = context.fit_stage(**cfg); packed = registry.stage(stage); stages.append(packed)
                for fi, f in enumerate(FRACTIONS):
                    record = stage['policies'][str(f)]; mid, value = registry.score(record['fit'])
                    selections['proposed'][4*mi+fi]['folds'].append({'model_id': mid, 'features': record['candidate']['features'], 'clean': value})
                mid, value = registry.score(stage['policies']['1.0']['fit'])
                selections['full'][mi]['folds'].append({'model_id': mid, 'features': list(range(X.shape[1])), 'clean': value})
            archived = native['splits'][i]['inner'][j]
            for method, grid in GRIDS.items():
                for ci, _ in enumerate(grid):
                    fs = sorted(archived['paths'][method][ci]['native']['features'])
                    for mi, cfg in enumerate(MODELS):
                        fit = context.classifier(fs, **cfg); mid, value = registry.score(fit)
                        selections[method][ci*len(MODELS)+mi]['folds'].append({'model_id': mid, 'features': fs, 'clean': value})
            inner_records.append({'reference_indices': a.tolist(), 'validation_indices': b.tolist(),
                'stages': stages, **registry.export(), 'native_path_source': {'outer_index': i, 'inner_index': j}})
            print(name, i, 'inner', j, 'models', len(registry.models), 'geometry', len(context.geometry), flush=True)
        choices = {method: choose(rows) for method, rows in selections.items()}
        context = GeometryStageContext(raw, labels); A, B = preprocess_fit(raw, X[vi])
        np.testing.assert_array_equal(A, context.A); registry = Registry(context)
        results = {}; outer_stages = {}; outer_paths = {}
        # Train every selected model before any original outer query label is used.
        for method in METHODS:
            chosen = selections[method][choices[method]]; cfg = MODELS[chosen['model_index']]
            geometry_id = None
            if method == 'proposed':
                stage = context.fit_stage(**cfg); outer_stages[method] = registry.stage(stage)
                record = stage['policies'][str(chosen['fraction'])]; fs = record['candidate']['features']; fit = record['fit']
                geometry_id = outer_stages[method]['policies'][str(chosen['fraction'])]['geometry_id']
            elif method == 'full':
                fs = list(range(X.shape[1])); fit = context.classifier(fs, **cfg)
            else:
                config = GRIDS[method][chosen['selector_index']]; key = json.dumps(config, sort_keys=True)
                cached = native['splits'][i]['outer_paths'][method]
                path = cached[key] if key in cached else PATHS[method](A, labels, **config)
                outer_paths[method] = {'config': config, 'path': path, 'cached': key in cached}
                fs = sorted(path['native']['features']); fit = context.classifier(fs, **cfg)
            results[method] = {'chosen_index': choices[method], 'model_index': chosen['model_index'],
                'selector_index': chosen['selector_index'], 'features': fs, 'model_id': registry.capture(fit),
                'geometry_id': geometry_id}
        rng = np.random.default_rng(144+i)
        queries = [B]+[np.clip(B+rng.uniform(-.01, .01, B.shape), 0, 1) for _ in range(3)]
        for r in results.values():
            fit = registry.models[r['model_id']]
            values = [balanced_accuracy(orthogonal_scores(Q, fit), fit['classes'], y[vi]) for Q in queries]
            r['metrics'] = {'clean': values[0], 'noise': float(np.mean(values[1:]))}
        records.append({'reference_indices': ri.tolist(), 'validation_indices': vi.tolist(),
            'inner': inner_records, 'trace': selections, 'results': results, 'stages': outer_stages,
            'native_paths': outer_paths, 'query_seed': 144+i, **registry.export()})
        payload = {'dataset': name, 'd': X.shape[1], 'splits': records, 'fingerprints': hashes,
            'complete': len(records) == len(source['splits']), 'teacher_guards_enforced': False,
            'scope': 'fresh_nested_FRS_geometry_with_common_classifier_exposed_development_not_confirmation'}
        out.write_text(json.dumps(payload, separators=(',', ':')), encoding='utf-8')
        print(name, i, 'outer complete', {m: r['metrics']['clean'] for m, r in results.items()}, flush=True)


def holm(values):
    ordered = sorted(values, key=values.get); result = {}; previous = 0.
    for k, name in enumerate(ordered):
        previous = max(previous, min(1., (len(values)-k)*values[name])); result[name] = previous
    return result


def summarize(names):
    rows = []; counts = {'outer_splits': 0, 'inner_splits': 0, 'models': 0, 'optimizer_caps': 0}
    for name in names:
        saved = json.loads((FOLDER/(name+'.json')).read_text()); assert saved['complete']
        splits = saved['splits']; methods = {}
        for method in METHODS:
            results = [s['results'][method] for s in splits]
            methods[method] = {metric: float(np.mean([r['metrics'][metric] for r in results])) for metric in ('clean', 'noise')}
            methods[method]['removed'] = float(np.mean([1-len(r['features'])/saved['d'] for r in results]))
        for s in splits:
            counts['outer_splits'] += 1; counts['inner_splits'] += len(s['inner'])
            for block in s['inner']+[s]:
                counts['models'] += len(block['models'])
                counts['optimizer_caps'] += sum(not m['optimizer_converged'] for m in block['models'])
        rows.append({'dataset': name, 'methods': methods})
    assert len(rows) == 15 and counts['outer_splits'] == 44 and counts['inner_splits'] == 128
    means = {m: {metric: float(np.mean([r['methods'][m][metric] for r in rows])) for metric in ('clean', 'noise', 'removed')} for m in METHODS}
    contrasts = {m: paired([r['methods']['proposed']['clean']-r['methods'][m]['clean'] for r in rows]) for m in METHODS[1:]}
    result = {'rows': rows, 'means': means, 'counts': counts, 'clean_contrasts': contrasts,
        'noise_contrasts': {m: paired([r['methods']['proposed']['noise']-r['methods'][m]['noise'] for r in rows]) for m in METHODS[1:]},
        'descriptive_holm_p': holm({m: c['sign_flip_two_sided_p'] for m, c in contrasts.items()}),
        'gate_passed': all(contrasts[m]['mean_pp'] >= .5 for m in ('fgs2023', 'sfrip2025')) and contrasts['full']['mean_pp'] >= 0
            and means['proposed']['removed'] >= .1 and counts['optimizer_caps']/counts['models'] <= .1,
        'scope': 'all15_fresh_nested_common_classifier_native_recent_controls_exposed_development_not_final_SOTA_confirmation',
        'limitations': 'No classifier-teacher guarantee; FGS/SFRIP independent implementations retain documented convention gaps; FD+/FCE+ fidelity unresolved.'}
    (FOLDER/'complete.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print({k: v for k, v in result.items() if k != 'rows'}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--only'); args = parser.parse_args()
    FOLDER.mkdir(exist_ok=True)
    names = [r['dataset'] for r in json.loads((SOURCE/'complete.json').read_text())['rows']]
    pending = [args.only] if args.only else names
    assert all(name in names for name in pending)
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run, name) for name in pending]): future.result()
    if not args.only: summarize(names)
