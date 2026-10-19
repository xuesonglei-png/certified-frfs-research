"""Independent pool contracts, fold selection, sklearn fits and predictions."""
import json
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_proper_loss_diverse_pool_pilot import ROOT, FOLDER, BASE, OOF, METHODS, METRICS, COUNTS, digest, verified_parent
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.nested_teacher_regularized import inner_splits
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.redistributed_loss import transfer_weights
from certified_frfs.capped_sparse import capped_sparse_projection
from certified_frfs.adaptive_multiknot_stage import MODEL_GRID
from audit_classwise_repair_admission import independent_class_certificate
from audit_flexible_majority_repair_admission import independent_protected, check_repair
from audit_safeguarded_crossfit_recent_pilot import check_model, classification
from audit_multiknot_forest_screen import independent_lp
from audit_forest_compact_transfer import equivalent
from review_box_iterations import paired
from run_nested_joint_path import holm


def expected_recipes(paths, ordinary, d):
    out = []
    for method in ('fgs2023', 'sfrip2025'):
        for si, p in enumerate(paths[method]):
            out.append({'features': sorted(p['ranking'][:int(np.ceil(.8*d))]), 'origin': {'method': method, 'config_index': si}})
    out.append({'features': sorted(ordinary), 'origin': {'method': 'OOF_ordinary_endpoint'}})
    out.append({'features': list(range(d)), 'origin': {'method': 'full_fallback'}})
    return out


def check_pool(A, y, recipes, pool, counts):
    d = A.shape[1]; full = np.ones(d)/np.sqrt(d)
    protected = independent_protected(A, y, full)
    expected = {}
    for r in recipes:
        expected.setdefault(tuple(r['features']), []).append(r['origin'])
    assert [tuple(r['features']) for r in pool] == list(expected)
    accepted_map = {}
    for record in pool:
        fs = record['features']; assert record['origins'] == expected[tuple(fs)]
        w = full if len(fs) == d else capped_sparse_projection(transfer_weights(A, fs, full)/np.sqrt(d), len(fs), 2/d)*np.sqrt(d)
        fresh = weight_certificate(A, y, fs, full, w, protected=protected)
        wide = np.zeros(d); wide[fs] = w
        equivalent(record['ordinary'], {'features': fs, 'weights': wide.tolist(), 'certificate': fresh})
        classes = independent_class_certificate(y, fresh)
        equivalent(classes, record['ordinary_class_certificate'])
        accepted = fresh['feasible'] and classes['feasible_all_classes']
        metric = record['ordinary']
        info = record['repair']
        if info is None:
            assert accepted
        else:
            assert not accepted and info['features'] == fs and info['time_limit'] == 10.
            counts['metric_solves'] += 1
            if 'weights' in info:
                fresh = weight_certificate(A, y, fs, full, np.array(info['weights']), protected=protected)
                equivalent(fresh, info['certificate'])
                classes = independent_class_certificate(y, fresh)
                equivalent(classes, info['classwise_dependency_certificate'])
                accepted = fresh['feasible'] and classes['feasible_all_classes']
                if fresh['feasible']:
                    check_repair(A, y, info, counts)
                wide = np.zeros(d); wide[fs] = info['weights']
                metric = {'features': fs, 'weights': wide.tolist(), 'certificate': fresh}
            else:
                assert info['status'] in ('no_incumbent', 'incumbent_constraints_failed', 'normalized_cap_failed')
                accepted = False
        assert record['accepted'] == accepted
        if accepted:
            equivalent(record['selected_metric'], metric)
            equivalent(record['selected_class_certificate'], classes)
        else:
            assert record['selected_metric'] is record['selected_class_certificate'] is None
        accepted_map[tuple(fs)] = bool(accepted)
        counts['FRS_pool_replays'] += 1
    assert accepted_map[tuple(range(d))]
    return accepted_map


def support_lists(recipes, accepted, paths, d, count):
    return {'proposed': [r['features'] if accepted[tuple(r['features'])] else list(range(d)) for r in recipes],
            'unguarded_pool': [r['features'] for r in recipes], 'full': [list(range(d))],
            'fgs_native': [sorted(p['native']['features']) for p in paths['fgs2023']],
            'sfrip_native': [sorted(p['native']['features']) for p in paths['sfrip2025']],
            'fgs_matched': [sorted(p['ranking'][:count]) for p in paths['fgs2023']],
            'sfrip_matched': [sorted(p['ranking'][:count]) for p in paths['sfrip2025']]}


def independent_choose(rows):
    losses = [float(np.mean([f['balanced_logloss'] for f in r['folds']])) for r in rows]
    means = [(float(np.mean([f['clean_BA'] for f in r['folds']])), float(np.mean([f['actual_count'] for f in r['folds']]))) for r in rows]
    for r, loss, (ba, size) in zip(rows, losses, means):
        equivalent(r['mean_balanced_logloss'], loss)
        equivalent(r['mean_clean_BA'], ba); equivalent(r['mean_actual_count'], size)
    return min(range(len(rows)), key=lambda i: (round(losses[i], 12), -round(means[i][0], 12), means[i][1], rows[i]['selector_index'], rows[i]['model_index']))


def audit(name):
    threadpool_limits(1)
    path = FOLDER/(name+'.json'); saved = json.loads(path.read_text())
    assert saved['complete'] and not saved['query_labels_used_for_selection']
    for rel, sha in saved['fingerprints'].items():
        assert digest(ROOT/rel) == sha, rel
    cached_path = FOLDER/(name+'-audit.json')
    if cached_path.exists():
        cached = json.loads(cached_path.read_text())
        if cached['artifact_sha256'] == digest(path) and cached['auditor_sha256'] == digest(Path(__file__)):
            assert cached['passed']
            return cached
    base, own = verified_parent(BASE, name), verified_parent(OOF, name)
    assert saved['reference_indices'] == base['reference_indices'] == own['reference_indices']
    assert saved['validation_indices'] == base['validation_indices'] == own['validation_indices']
    X, y = load_dataset(name)
    ri, vi = np.array(saved['reference_indices']), np.array(saved['validation_indices'])
    assert not set(ri)&set(vi)
    raw, labels, d = X[ri], y[ri], saved['d']
    splits = inner_splits(labels, seed=20261033)
    assert len(saved['inner']) == len(splits)
    counts = Counter(datasets=1)
    selected = saved['selection']['selected']; combined = saved['selection']['options']
    # Selection replay BEFORE using any outer labels.
    for m in METHODS:
        assert len(combined[m]) == COUNTS[m]*len(MODEL_GRID)
        for i, row in enumerate(combined[m]):
            assert row['selector_index'] == i//len(MODEL_GRID) and row['model_index'] == i%len(MODEL_GRID)
            equivalent(row['folds'], [b['options'][m][i]['folds'][0] for b in saved['inner']])
        assert selected[m] == independent_choose(combined[m])
    recipe_index = combined['proposed'][selected['proposed']]['selector_index']
    outer_metrics = {}
    for fi, block in enumerate(saved['inner']+[saved['outer']]):
        outer = fi == len(splits)
        if outer:
            train_raw, train_y, query_raw, query_y = raw, labels, X[vi], y[vi]
            parent_block, ordinary_block = base['outer'], own['outer']
        else:
            ca, cb = splits[fi]
            assert block['reference_indices'] == ca.tolist() and block['validation_indices'] == cb.tolist()
            assert not set(ca)&set(cb)
            train_raw, train_y, query_raw, query_y = raw[ca], labels[ca], raw[cb], labels[cb]
            parent_block, ordinary_block = base['inner'][fi], own['inner'][fi]
        A, B = preprocess_fit(train_raw, query_raw)
        equivalent(block['paths'], parent_block['paths'])
        recipes = expected_recipes(parent_block['paths'], ordinary_block['stage']['candidate']['features'], d)
        equivalent(block['recipes'], recipes)
        accepted = check_pool(A, train_y, recipes, block['pool'], counts)
        count = len(recipes[recipe_index]['features']) if accepted[tuple(recipes[recipe_index]['features'])] else d
        assert block['matched_proposed_count'] == count
        support_map = support_lists(recipes, accepted, parent_block['paths'], d, count)
        for fit in block['models']:
            check_model(A, train_y, fit, counts)
            counts['downstream_optimizer_caps'] += not fit['optimizer_converged']
        if not outer:
            evaluations = {str(i): classification(B, query_y, fit) for i, fit in enumerate(block['models'])}
            equivalent(block['evaluations'], evaluations)
            for m in METHODS:
                rows = block['options'][m]
                assert len(rows) == COUNTS[m]*len(MODEL_GRID)
                for i, row in enumerate(rows):
                    si, mi = i//len(MODEL_GRID), i%len(MODEL_GRID)
                    assert row['selector_index'] == si and row['model_index'] == mi
                    fs = support_map[m][si]; fold = row['folds'][0]; mid = fold['model_id']; fit = block['models'][mid]
                    assert len(row['folds']) == 1 and fit['features'] == fs
                    assert fit['C'] == MODEL_GRID[mi]['C'] and len(fit['knots']) == MODEL_GRID[mi]['knot_count']
                    equivalent(fold, {'model_id': mid, 'features': fs, 'actual_count': len(fs), 'clean_BA': evaluations[str(mid)]['metrics']['clean_BA'], 'balanced_logloss': evaluations[str(mid)]['metrics']['balanced_logloss']})
                    counts['inner_option_replays'] += 1
            counts['inner_folds'] += 1
        else:
            for m in METHODS:
                result = block['results'][m]; option = combined[m][selected[m]]
                assert result['option_index'] == selected[m] and result['selector_index'] == option['selector_index'] and result['model_index'] == option['model_index']
                fs = support_map[m][option['selector_index']]; fit = block['models'][result['model_id']]
                assert result['features'] == fit['features'] == fs
                mi = option['model_index']
                assert fit['C'] == MODEL_GRID[mi]['C'] and len(fit['knots']) == MODEL_GRID[mi]['knot_count']
                report = classification(B, query_y, fit, certify=True)
                equivalent(result['evaluation'], report)
                own_class = int(np.searchsorted(fit['classes'], query_y[0])); rival = int(own_class == 0)
                lp = independent_lp(B[0], own_class, rival, fit)
                np.testing.assert_allclose(lp, report['pair_margins'][0][rival], atol=1e-7)
                outer_metrics[m] = report['metrics']
                counts['outer_prediction_replays'] += 1
            assert len(block['results']['proposed']['features']) == len(block['results']['fgs_matched']['features']) == len(block['results']['sfrip_matched']['features'])
        print(name, 'outer' if outer else 'inner'+str(fi), 'pool/selection/fits audited', flush=True)
    sizes = {m: len(saved['outer']['results'][m]['features']) for m in METHODS}
    row = {'dataset': name, 'metrics': outer_metrics, 'actual_counts': sizes, 'd': d, 'removed': 1-sizes['proposed']/d}
    output = {'row': row, 'counts': dict(counts), 'artifact_sha256': digest(path), 'passed': True,
              'auditor_sha256': digest(Path(__file__))}
    cached_path.write_text(json.dumps(output, indent=2), encoding='utf-8')
    return output


if __name__ == '__main__':
    summary = json.loads((FOLDER/'complete.json').read_text())
    reports, counts = [], Counter()
    with ProcessPoolExecutor(max_workers=2) as executor:
        for future in as_completed([executor.submit(audit, r['dataset']) for r in summary['rows']]):
            report = future.result(); reports.append(report); counts.update(report['counts'])
    rows = sorted([r['row'] for r in reports], key=lambda r: r['dataset'])
    equivalent(rows, sorted(summary['rows'], key=lambda r: r['dataset']))
    means = {m: {k: float(np.mean([r['metrics'][m][k] for r in rows])) for k in METRICS} for m in METHODS}
    equivalent(means, summary['means'])
    contrasts = {m: paired([r['metrics']['proposed']['clean_BA']-r['metrics'][m]['clean_BA'] for r in rows]) for m in METHODS if m != 'proposed'}
    equivalent(contrasts, summary['contrasts_clean_BA'])
    primary = ('fgs_matched', 'sfrip_matched')
    equivalent(holm({m: contrasts[m]['sign_flip_two_sided_p'] for m in primary}), summary['primary_descriptive_holm'])
    removed = float(np.mean([r['removed'] for r in rows])); equivalent(removed, summary['mean_removed'])
    gate = all(contrasts[m]['mean_pp'] >= .5 and contrasts[m]['wins'] >= 10 for m in primary)
    gate = gate and contrasts['full']['mean_pp'] >= 0 and removed >= .1 and counts['downstream_optimizer_caps'] == 0
    assert summary['pilot_gate_passed'] == gate and not summary['goal_achieved']
    assert counts['datasets'] == 15 and counts['inner_folds'] == 44
    assert counts['independent_training_fits'] == summary['counts']['downstream_unique_fits']
    assert counts['metric_solves'] == summary['counts']['metric_solves']
    output = {'passed': True, 'counts': dict(counts), 'artifact_hashes': {r['row']['dataset']: r['artifact_sha256'] for r in reports},
              'summary_sha256': digest(FOLDER/'complete.json'), 'auditor_sha256': digest(Path(__file__)),
              'goal_achieved': False, 'scope': 'independent_pool_contract_class_means_sklearn_fold_selection_and_outer_replay_not_confirmation'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(output, flush=True)
