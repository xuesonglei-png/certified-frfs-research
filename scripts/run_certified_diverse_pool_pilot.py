"""Fold-local certified candidate recipes versus an equal-budget raw pool."""
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_recent_pool_certificate_admission import ROOT, BASE, OOF, digest, verified_parent
from run_safeguarded_crossfit_recent_pilot import Registry, metric_scores, choose, METRICS
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.geometry_only_forest_stage import GeometryStageContext
from certified_frfs.safeguarded_crossfit_stage import safeguarded_metric_repair
from certified_frfs.classwise_dependency_certificate import classwise_dependency_certificate
from certified_frfs.nested_teacher_regularized import inner_splits
from certified_frfs.adaptive_multiknot_stage import MODEL_GRID
from review_box_iterations import paired
from run_nested_joint_path import holm

FOLDER = ROOT/'results/certified-diverse-pool-pilot'
METHODS = ('proposed', 'unguarded_pool', 'full', 'fgs_native', 'sfrip_native', 'fgs_matched', 'sfrip_matched')
COUNTS = {'proposed': 17, 'unguarded_pool': 17, 'full': 1, 'fgs_native': 9, 'sfrip_native': 6, 'fgs_matched': 9, 'sfrip_matched': 6}


def options(method):
    return [{'selector_index': si, 'model_index': mi, 'folds': []}
            for si in range(COUNTS[method]) for mi in range(len(MODEL_GRID))]


def candidate_origins(paths, ordinary, d):
    target = max(1, int(np.ceil(.8*d)))
    recipes = []
    for method in ('fgs2023', 'sfrip2025'):
        for i, path in enumerate(paths[method]):
            recipes.append({'features': sorted(path['ranking'][:target]), 'origin': {'method': method, 'config_index': i}})
    recipes += [{'features': sorted(ordinary), 'origin': {'method': 'OOF_ordinary_endpoint'}},
                {'features': list(range(d)), 'origin': {'method': 'full_fallback'}}]
    assert len(recipes) == 17
    return recipes


def admit(raw, labels, recipes):
    ctx = GeometryStageContext(raw, labels)
    pool = {}
    for recipe in recipes:
        pool.setdefault(tuple(recipe['features']), []).append(recipe['origin'])
    records = []
    for fs, origins in pool.items():
        start = time.perf_counter()
        ordinary = ctx.candidate(fs)
        classes = classwise_dependency_certificate(labels, ordinary['certificate'])
        accepted = ordinary['certificate']['feasible'] and classes['feasible_all_classes']
        metric, repair = ordinary, None
        if not accepted:
            weights, repair = safeguarded_metric_repair(ctx.A, labels, list(fs), time_limit=10.)
            if weights is not None:
                wide = np.zeros(ctx.d); wide[list(fs)] = weights
                metric = {'features': list(fs), 'weights': wide.tolist(), 'certificate': repair['certificate']}
                classes = repair['classwise_dependency_certificate']
                accepted = True
        records.append({'features': list(fs), 'origins': origins, 'ordinary': ordinary,
                        'ordinary_class_certificate': classwise_dependency_certificate(labels, ordinary['certificate']),
                        'repair': repair, 'accepted': bool(accepted), 'selected_metric': metric if accepted else None,
                        'selected_class_certificate': classes if accepted else None,
                        'wall_seconds': time.perf_counter()-start})
    assert any(r['accepted'] and len(r['features']) == ctx.d for r in records)
    return records


def supports(recipes, pool, paths, d, *, count=None):
    accepted = {tuple(r['features']): r['accepted'] for r in pool}
    output = {'proposed': [r['features'] if accepted[tuple(r['features'])] else list(range(d)) for r in recipes],
              'unguarded_pool': [r['features'] for r in recipes], 'full': [list(range(d))],
              'fgs_native': [sorted(p['native']['features']) for p in paths['fgs2023']],
              'sfrip_native': [sorted(p['native']['features']) for p in paths['sfrip2025']]}
    if count is not None:
        output.update({'fgs_matched': [sorted(p['ranking'][:count]) for p in paths['fgs2023']],
                       'sfrip_matched': [sorted(p['ranking'][:count]) for p in paths['sfrip2025']]})
    return output


def fingerprint(name):
    files = [__file__, ROOT/'research/certified-diverse-pool-pilot-protocol.md',
             ROOT/'scripts/run_safeguarded_crossfit_recent_pilot.py', ROOT/'scripts/run_recent_pool_certificate_admission.py',
             BASE/(name+'.json'), BASE/'complete-audit.json', OOF/(name+'.json'), OOF/'complete-audit.json',
             ROOT/'results/recent-pool-certificate-admission/complete-audit.json',
             ROOT/'results/recent-pool-certificate-admission'/(name+'.json'), ROOT/'data/processed'/(name+'.npz')]
    files += sorted((ROOT/'src/certified_frfs').glob('*.py'))
    return {str(p.relative_to(ROOT).as_posix()): digest(p) for p in map(type(ROOT), files)}


def score_options(registry, method, support_list):
    rows = options(method)
    for row in rows:
        fs = support_list[row['selector_index']]
        mid = registry.fit(fs, row['model_index'])
        row['folds'] = [{'model_id': mid, 'features': sorted(fs), 'actual_count': len(fs),
                         'clean_BA': registry.evaluations[str(mid)]['metrics']['clean_BA']}]
    return rows


def run(name):
    threadpool_limits(1)
    hashes = fingerprint(name)
    out = FOLDER/(name+'.json')
    if out.exists():
        old = json.loads(out.read_text())
        assert old['fingerprints'] == hashes
        if old['complete']:
            return
    base, own = verified_parent(BASE, name), verified_parent(OOF, name)
    X, y = load_dataset(name)
    ri, vi = np.array(base['reference_indices']), np.array(base['validation_indices'])
    assert base['reference_indices'] == own['reference_indices'] and base['validation_indices'] == own['validation_indices']
    raw, labels = X[ri], y[ri]
    inners, registries = [], []
    for fi, (ca, cb) in enumerate(inner_splits(labels, seed=20261033)):
        for parent in (base, own):
            assert parent['inner'][fi]['reference_indices'] == ca.tolist()
            assert parent['inner'][fi]['validation_indices'] == cb.tolist()
        paths = base['inner'][fi]['paths']
        recipes = candidate_origins(paths, own['inner'][fi]['stage']['candidate']['features'], X.shape[1])
        start = time.perf_counter()
        pool = admit(raw[ca], labels[ca], recipes)
        A, B = preprocess_fit(raw[ca], raw[cb])
        registry = Registry(A, labels[ca], B, labels[cb])
        fsmap = supports(recipes, pool, paths, X.shape[1])
        rows = {m: score_options(registry, m, fsmap[m]) for m in fsmap}
        inners.append({'reference_indices': ca.tolist(), 'validation_indices': cb.tolist(), 'recipes': recipes,
                       'pool': pool, 'paths': paths, 'options': rows, 'stage_and_fit_seconds': time.perf_counter()-start})
        registries.append(registry)
        print(name, 'inner', fi, 'admitted', sum(r['accepted'] for r in pool), '/', len(pool), 'fits', len(registry.models), flush=True)
    combined, selected = {}, {}
    for m in METHODS[:5]:
        combined[m] = options(m)
        for i, row in enumerate(combined[m]):
            row['folds'] = [b['options'][m][i]['folds'][0] for b in inners]
        selected[m] = choose(combined[m])
    chosen_recipe = combined['proposed'][selected['proposed']]['selector_index']
    for block, registry in zip(inners, registries):
        count = len(supports(block['recipes'], block['pool'], block['paths'], X.shape[1])['proposed'][chosen_recipe])
        block['matched_proposed_count'] = count
        fsmap = supports(block['recipes'], block['pool'], block['paths'], X.shape[1], count=count)
        for m in METHODS[5:]:
            block['options'][m] = score_options(registry, m, fsmap[m])
        block.update(registry.export())
    for m in METHODS[5:]:
        combined[m] = options(m)
        for i, row in enumerate(combined[m]):
            row['folds'] = [b['options'][m][i]['folds'][0] for b in inners]
        selected[m] = choose(combined[m])
    paths = base['outer']['paths']
    recipes = candidate_origins(paths, own['outer']['stage']['candidate']['features'], X.shape[1])
    admission_folder = ROOT/'results/recent-pool-certificate-admission'
    audit = json.loads((admission_folder/'complete-audit.json').read_text())
    assert audit['passed'] and audit['artifact_hashes'][name] == digest(admission_folder/(name+'.json'))
    saved_pool = json.loads((admission_folder/(name+'.json')).read_text())
    assert saved_pool['reference_indices'] == ri.tolist()
    pool = saved_pool['pool']
    count = len(supports(recipes, pool, paths, X.shape[1])['proposed'][chosen_recipe])
    fsmap = supports(recipes, pool, paths, X.shape[1], count=count)
    A, B = preprocess_fit(raw, X[vi])
    registry = Registry(A, labels)
    results = {}
    for m in METHODS:
        option = combined[m][selected[m]]
        fs = fsmap[m][option['selector_index']]
        mid = registry.fit(fs, option['model_index'])
        results[m] = {'option_index': selected[m], 'selector_index': option['selector_index'], 'model_index': option['model_index'],
                      'features': fs, 'model_id': mid, 'evaluation': metric_scores(B, y[vi], registry.models[mid], certify=True)}
    assert hashes == fingerprint(name)
    output = {'dataset': name, 'd': X.shape[1], 'reference_indices': ri.tolist(), 'validation_indices': vi.tolist(),
              'fingerprints': hashes, 'inner': inners, 'selection': {'options': combined, 'selected': selected},
              'outer': {'recipes': recipes, 'pool': pool, 'paths': paths, 'matched_proposed_count': count, 'results': results, **registry.export()},
              'complete': True, 'query_labels_used_for_selection': False, 'independent_audit_required': True,
              'scope': 'exposed_development_fold_local_certified_and_equal_budget_unguarded_pool_not_confirmation'}
    out.write_text(json.dumps(output, separators=(',', ':')), encoding='utf-8')
    print(name, 'COMPLETE', {m: round(results[m]['evaluation']['metrics']['clean_BA'], 6) for m in METHODS}, flush=True)


def summarize(names):
    rows, counts = [], Counter()
    for name in names:
        saved = json.loads((FOLDER/(name+'.json')).read_text())
        assert saved['complete'] and not saved['query_labels_used_for_selection']
        results = saved['outer']['results']
        sizes = {m: len(results[m]['features']) for m in METHODS}
        assert sizes['proposed'] == sizes['fgs_matched'] == sizes['sfrip_matched']
        metrics = {m: results[m]['evaluation']['metrics'] for m in METHODS}
        rows.append({'dataset': name, 'metrics': metrics, 'actual_counts': sizes, 'd': saved['d'], 'removed': 1-sizes['proposed']/saved['d']})
        for block in saved['inner']+[saved['outer']]:
            counts['downstream_unique_fits'] += len(block['models'])
            counts['downstream_optimizer_caps'] += sum(not f['optimizer_converged'] for f in block['models'])
            counts['metric_solves'] += sum(r['repair'] is not None for r in block['pool'])
        counts['datasets'] += 1
        counts['inner_folds'] += len(saved['inner'])
    means = {m: {k: float(np.mean([r['metrics'][m][k] for r in rows])) for k in METRICS} for m in METHODS}
    contrasts = {m: paired([r['metrics']['proposed']['clean_BA']-r['metrics'][m]['clean_BA'] for r in rows]) for m in METHODS if m != 'proposed'}
    primary = ('fgs_matched', 'sfrip_matched')
    removed = float(np.mean([r['removed'] for r in rows]))
    gate = all(contrasts[m]['mean_pp'] >= .5 and contrasts[m]['wins'] >= 10 for m in primary)
    gate = gate and contrasts['full']['mean_pp'] >= 0 and removed >= .1 and counts['downstream_optimizer_caps'] == 0
    report = {'rows': rows, 'means': means, 'counts': dict(counts), 'contrasts_clean_BA': contrasts,
              'primary_descriptive_holm': holm({m: contrasts[m]['sign_flip_two_sided_p'] for m in primary}),
              'mean_removed': removed, 'pilot_gate_passed': bool(gate), 'goal_achieved': False, 'independent_audit_required': True,
              'scope': 'all15_exposed_development_equal_recipe_budget_not_confirmation_or_FD_FCE'}
    (FOLDER/'complete.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print({k: v for k, v in report.items() if k != 'rows'}, flush=True)


if __name__ == '__main__':
    FOLDER.mkdir(exist_ok=True)
    names = sorted(json.loads((BASE/'complete-audit.json').read_text())['artifact_hashes'])
    with ProcessPoolExecutor(max_workers=2) as executor:
        for result in as_completed([executor.submit(run, name) for name in names]):
            result.result()
    summarize(names)
