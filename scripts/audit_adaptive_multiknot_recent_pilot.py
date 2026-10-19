"""Independent sklearn design, original baseline paths and raw-loss audit."""
import argparse
import json
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.linalg import helmert
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from run_nested_joint_path import ROOT, digest, holm
from run_adaptive_multiknot_recent_pilot import FOLDER, METHODS, METRICS
from compare_trajectory_recent import GRIDS
from certified_frfs.adaptive_multiknot_stage import MODEL_GRID
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.nested_teacher_regularized import inner_splits
from certified_frfs.fuzzy_interaction_tree import interaction_tree
from certified_frfs.recent_budget import fgs_budget_path, sfrip_budget_path
from certified_frfs.redistributed_loss import transfer_weights
from certified_frfs.capped_sparse import capped_sparse_projection
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.majority_metric import MajorityTerms
from certified_frfs.multiknot_forest import box_margins
from audit_multiknot_forest_screen import independent_scores, independent_lp
from audit_forest_compact_transfer import equivalent
from review_box_iterations import paired


def fresh_model(A, y, fs, knots, C, pair_penalty, edges):
    """Independent nodal interpolation/design/fit/decoding implementation."""
    classes, counts = np.unique(y, return_counts=True)
    q, d, m = len(classes), A.shape[1], len(knots)
    W, V = np.zeros((q, d, m)), np.zeros((q, len(edges), m, m))
    if not fs:
        return {'coef': W, 'pair_coef': V, 'intercept': np.log(counts/counts.sum()),
                'classes': classes, 'optimizer_converged': True, 'optimizer_iterations': 0}
    hats = np.stack([np.interp(A, knots, col) for col in np.eye(m).T], axis=-1)
    contrast = helmert(m, full=False).T
    coordinates = hats@contrast
    blocks = [coordinates[:, fs].reshape(len(A), -1)]
    r = m-1
    for u, v in edges:
        blocks.append((coordinates[:, u, :, None]*coordinates[:, v, None, :]).reshape(len(A), r*r)/np.sqrt(pair_penalty))
    learner = LogisticRegression(C=C, class_weight='balanced', max_iter=4000, tol=1e-8, random_state=0).fit(np.column_stack(blocks), y)
    coef, bias = learner.coef_, learner.intercept_
    if q == 2:
        coef = np.vstack([np.zeros_like(coef[0]), coef[0]])
        bias = np.r_[0., bias[0]]
    W[:, fs] = coef[:, :r*len(fs)].reshape(q, len(fs), r)@contrast.T
    for e in range(len(edges)):
        tensor = coef[:, r*len(fs)+r*r*e:r*len(fs)+r*r*(e+1)].reshape(q, r, r)/np.sqrt(pair_penalty)
        V[:, e] = np.einsum('ki,qij,lj->qkl', contrast, tensor, contrast)
    W -= W[:1].copy()
    V -= V[:1].copy()
    bias -= bias[:1].copy()
    return {'coef': W, 'pair_coef': V, 'intercept': bias, 'classes': classes,
            'optimizer_converged': int(learner.n_iter_.max()) < learner.max_iter,
            'optimizer_iterations': int(learner.n_iter_.max())}


def check_model(A, y, fit, counts):
    fs, edges, knots = fit['features'], fit['edges'], fit['knots']
    assert fs == sorted(set(fs)) and fit['classes'] == np.unique(y).tolist()
    assert fit['pair_penalty'] == 1. and fit['C'] in (.1, 1., 10.) and not fit['teacher_guards_enforced']
    assert len(knots) in (3, 5)
    np.testing.assert_array_equal(knots, np.linspace(0, 1, len(knots)))
    if fs:
        local = interaction_tree(A[:, fs], y)
        assert edges == [sorted([fs[u], fs[v]]) for u, v in local['edges']]
        assert fit['class_balanced']
    else:
        assert not edges and not fit['class_balanced'] and 'training_prior' in fit['scope']
    fresh = fresh_model(A, y, fs, np.array(knots), fit['C'], 1., edges)
    for key in ('coef', 'pair_coef', 'intercept'):
        # JSON serializes a zero-edge tensor as [] under each class.
        expected = np.array(fit[key])
        if key == 'pair_coef':
            expected = expected.reshape(fresh[key].shape)
        np.testing.assert_allclose(expected, fresh[key], atol=1e-8, rtol=1e-8)
    assert fresh['optimizer_converged'] == fit['optimizer_converged']
    assert fresh['optimizer_iterations'] == fit['optimizer_iterations']
    counts['independent_training_fits'] += 1
    return fresh


def classification(B, truth, fit, certify=False):
    value = independent_scores(B, fit)
    classes = np.array(fit['classes'])
    own = np.searchsorted(classes, truth)
    pred = classes[value.argmax(axis=1)]
    loss = logsumexp(value, axis=1)-value[np.arange(len(B)), own]
    def balanced(array):
        return float(np.mean([np.mean(array[truth == c]) for c in np.unique(truth)]))
    metrics = {'clean_BA': balanced(pred == truth), 'balanced_logloss': balanced(loss),
               'macro_F1': float(f1_score(truth, pred, average='macro', zero_division=0))}
    result = {'scores': value.tolist(), 'predictions': pred.tolist(), 'metrics': metrics}
    if certify:
        oracle = box_margins(B, truth, fit, eps=.01, witnesses=True)
        pair = oracle['pair_margins'].copy()
        pair[np.arange(len(B)), own] = 0.
        result['pair_margins'] = pair.tolist()
        metrics['certified_BA'] = balanced(oracle['margins'] > 1e-10)
        witness = independent_scores(oracle['witness_raw'], fit)
        np.testing.assert_allclose(witness[np.arange(len(B)), own]-witness[np.arange(len(B)), oracle['rivals']], oracle['margins'], atol=1e-9)
    return result


def direct_projection(A, y, teacher, center, features):
    raw = A.copy()
    omitted = [a for a in range(A.shape[1]) if a not in features]
    raw[:, omitted] = center[omitted]
    difference = independent_scores(A, teacher)-independent_scores(raw, teacher)
    difference -= difference.mean(axis=1, keepdims=True)
    weights = np.array([1/(len(np.unique(y))*sum(y == c)) for c in y])
    return float(weights@np.mean(difference**2, axis=1))


def check_stage(raw, y, stage, counts):
    A, _ = preprocess_fit(raw, raw)
    d, mass = A.shape[1], np.sqrt(A.shape[1])
    fullweights = np.ones(d)/mass
    protected = MajorityTerms(A, y, weights=fullweights).bounds(range(d), allowed_enemies=2) > 1e-10
    selection = stage['teacher_selection']
    splits = inner_splits(y, seed=20261032)
    assert len(splits) == len(selection['folds'])
    losses, caps = [], 0
    for (a, b), block in zip(splits, selection['folds']):
        assert block['reference_indices'] == a.tolist() and block['validation_indices'] == b.tolist()
        assert set(a).isdisjoint(b) and len(set(a)|set(b)) == len(raw)
        F, Q = preprocess_fit(raw[a], raw[b])
        equivalent(block['tree'], interaction_tree(F, y[a]))
        row = []
        assert len(block['models']) == len(MODEL_GRID)
        for config, fit in zip(MODEL_GRID, block['models']):
            assert fit['features'] == list(range(d)) and fit['C'] == config['C'] and len(fit['knots']) == config['knot_count']
            check_model(F, y[a], fit, counts)
            caps += not fit['optimizer_converged']
            row.append(classification(Q, y[b], fit)['metrics']['balanced_logloss'])
        equivalent(row, block['balanced_logloss'])
        losses.append(row)
    options = [{'config': config, 'mean_balanced_logloss': float(np.mean([v[i] for v in losses]))}
               for i, config in enumerate(MODEL_GRID)]
    equivalent(options, selection['options'])
    chosen = min(range(len(options)), key=lambda i: (round(options[i]['mean_balanced_logloss'], 12), i))
    assert chosen == selection['chosen_index']
    teacher = stage['teacher']
    assert teacher['C'] == MODEL_GRID[chosen]['C'] and len(teacher['knots']) == MODEL_GRID[chosen]['knot_count']
    assert teacher['features'] == list(range(d))
    equivalent(stage['tree'], interaction_tree(A, y))
    check_model(A, y, teacher, counts)
    caps += not teacher['optimizer_converged']
    assert caps == selection['optimizer_caps']
    alpha = np.array([1/(len(np.unique(y))*sum(y == c)) for c in y])
    center = alpha@A
    np.testing.assert_allclose(center, stage['path']['center'], atol=1e-12)
    path, active, offset, accepted_records = stage['path'], list(range(d)), 0, []
    cache = {}
    def candidate(fs):
        key = tuple(fs)
        if key not in cache:
            weights = fullweights if len(fs) == d else capped_sparse_projection(transfer_weights(A, fs, fullweights)/mass, len(fs), 2/d)*mass
            embedded = np.zeros(d)
            embedded[fs] = weights
            cert = weight_certificate(A, y, fs, fullweights, weights, protected=protected)
            cache[key] = {'features': fs, 'weights': embedded.tolist(), 'certificate': cert,
                          'projection_loss': direct_projection(A, y, teacher, center, fs)}
            counts['FRS_certificate_replays'] += 1
        return cache[key]
    accepted_records.append(candidate(active))
    assert stage['fraction'] == .8 and stage['target_count'] == max(1, int(np.ceil(.8*d)))
    assert path['tie_decimals'] == 12
    for entry in path['rankings']:
        assert entry['features'] == active and len(active) > stage['target_count']
        ranking = [(direct_projection(A, y, teacher, center, [a for a in active if a != r]), r) for r in active]
        ranking.sort(key=lambda item: (round(item[0], 12), item[1]))
        equivalent(entry['choices'], [{'removed': int(r), 'loss': value} for value, r in ranking])
        accepted = None
        for loss, removed in ranking:
            record = path['attempts'][offset]
            offset += 1
            fs = [a for a in active if a != removed]
            expected = {**candidate(fs), 'removed': int(removed), 'accepted': bool(candidate(fs)['certificate']['feasible'])}
            equivalent(record, expected)
            if expected['accepted']:
                accepted = candidate(fs)
                accepted_records.append(accepted)
                break
        if accepted is None:
            assert entry is path['rankings'][-1]
            break
        active = accepted['features']
    assert offset == len(path['attempts'])
    equivalent(path['trace'], accepted_records)
    equivalent(stage['candidate'], accepted_records[-1])
    assert stage['actual_count'] == len(active) and stage['candidate']['certificate']['feasible']
    assert not stage['teacher_guards_enforced']
    counts['stages'] += 1


def audit(name):
    threadpool_limits(1)
    artifact = FOLDER/(name+'.json')
    saved = json.loads(artifact.read_text())
    assert saved['complete'] and not saved['query_labels_used_for_selection']
    fp = {'artifact_sha256': digest(artifact), 'auditor_sha256': digest(Path(__file__))}
    checkpoint = FOLDER/(name+'-audit.json')
    if checkpoint.exists():
        old = json.loads(checkpoint.read_text())
        if old['fingerprints'] == fp:
            return old['result']
    for rel, sha in saved['fingerprints'].items():
        assert digest(ROOT/rel) == sha, rel
    X, y = load_dataset(name)
    assert saved['raw_n'] == len(X) and saved['d'] == X.shape[1]
    ri, vi = np.array(saved['reference_indices']), np.array(saved['validation_indices'])
    expected_ri, expected_vi = list(StratifiedKFold(3, shuffle=True, random_state=101).split(X, y))[0]
    np.testing.assert_array_equal(ri, expected_ri)
    np.testing.assert_array_equal(vi, expected_vi)
    assert set(ri).isdisjoint(vi) and len(ri)+len(vi) == len(X)
    raw, labels = X[ri], y[ri]
    splits = inner_splits(labels, seed=20261033)
    assert len(splits) == len(saved['inner'])
    counts = Counter()
    outer_metrics = None
    for index, block in enumerate(saved['inner']+[saved['outer']]):
        outer = index == len(saved['inner'])
        if outer:
            ref, train, query, truth = raw, labels, X[vi], y[vi]
        else:
            ca, cb = splits[index]
            assert block['reference_indices'] == ca.tolist() and block['validation_indices'] == cb.tolist()
            assert set(ca).isdisjoint(cb) and len(set(ca)|set(cb)) == len(raw)
            ref, train, query, truth = raw[ca], labels[ca], raw[cb], labels[cb]
        A, B = preprocess_fit(ref, query)
        check_stage(ref, train, block['stage'], counts)
        for method, configs, original in [('fgs2023', GRIDS['fgs2023'], fgs_budget_path),
                                         ('sfrip2025', GRIDS['sfrip2025'], sfrip_budget_path)]:
            assert len(block['paths'][method]) == len(configs)
            for config, path in zip(configs, block['paths'][method]):
                fresh = original(A, train, **config)
                equivalent(path, fresh)
                assert sorted(path['ranking']) == list(range(A.shape[1]))
                counts['original_baseline_paths_rebuilt'] += 1
        for tree in block['trees']:
            fs = tree['features']
            equivalent(tree['local_tree'], interaction_tree(A[:, fs], train))
            assert tree['edges'] == [sorted([fs[u], fs[v]]) for u, v in tree['local_tree']['edges']]
        for mid, fit in enumerate(block['models']):
            check_model(A, train, fit, counts)
            counts['downstream_unique_fits'] += 1
            counts['downstream_optimizer_caps'] += not fit['optimizer_converged']
            if not outer:
                equivalent(block['evaluations'][str(mid)], classification(B, truth, fit))
                counts['inner_prediction_replays'] += 1
        counts['teacher_optimizer_caps'] += block['stage']['teacher_selection']['optimizer_caps']
        counts['teacher_fits'] += 1+sum(len(f['models']) for f in block['stage']['teacher_selection']['folds'])
        count = block['stage']['actual_count']
        support_lists = {'proposed': [block['stage']['candidate']['features']], 'full': [list(range(A.shape[1]))],
                         'fgs_native': [p['native']['features'] for p in block['paths']['fgs2023']],
                         'fgs_matched': [p['ranking'][:count] for p in block['paths']['fgs2023']],
                         'sfrip_native': [p['native']['features'] for p in block['paths']['sfrip2025']],
                         'sfrip_matched': [p['ranking'][:count] for p in block['paths']['sfrip2025']]}
        if not outer:
            for m in METHODS:
                options = block['options'][m]
                assert len(options) == len(support_lists[m])*len(MODEL_GRID)
                for i, row in enumerate(options):
                    si, mi = divmod(i, len(MODEL_GRID))
                    assert row['selector_index'] == si and row['model_index'] == mi
                    report = row['folds'][0]
                    fs = sorted(support_lists[m][si])
                    model = block['models'][report['model_id']]
                    assert report['features'] == fs == model['features'] and report['actual_count'] == len(fs)
                    assert model['C'] == MODEL_GRID[mi]['C'] and len(model['knots']) == MODEL_GRID[mi]['knot_count']
                    equivalent(report['clean_BA'], block['evaluations'][str(report['model_id'])]['metrics']['clean_BA'])
                    if m in ('fgs_matched', 'sfrip_matched'):
                        assert len(fs) == count
                        counts['inner_count_options_verified'] += 1
                    counts['inner_classifier_options_verified'] += 1
        else:
            outer_metrics = {}
            for m in METHODS:
                selected = saved['selection']['selected'][m]
                combined = saved['selection']['options'][m]
                for i, option in enumerate(combined):
                    expected_folds = [inner['options'][m][i]['folds'][0] for inner in saved['inner']]
                    assert option['folds'] == expected_folds
                    equivalent(option['mean_clean_BA'], float(np.mean([v['clean_BA'] for v in expected_folds])))
                    equivalent(option['mean_actual_count'], float(np.mean([v['actual_count'] for v in expected_folds])))
                chosen = min(range(len(combined)), key=lambda i: (-round(combined[i]['mean_clean_BA'], 12),
                    combined[i]['mean_actual_count'], combined[i]['selector_index'], combined[i]['model_index']))
                assert selected == chosen
                result = block['results'][m]
                assert result['option_index'] == chosen and result['selector_index'] == combined[chosen]['selector_index']
                assert result['model_index'] == combined[chosen]['model_index']
                fs = sorted(support_lists[m][result['selector_index']])
                fit = block['models'][result['model_id']]
                assert fs == result['features'] == fit['features']
                config = MODEL_GRID[result['model_index']]
                assert fit['C'] == config['C'] and len(fit['knots']) == config['knot_count']
                report = classification(B, truth, fit, certify=True)
                equivalent(result['evaluation'], report)
                outer_metrics[m] = report['metrics']
                classes = np.array(fit['classes'])
                own = int(np.searchsorted(classes, truth[0]))
                rival = 1 if own == 0 else 0
                lp = independent_lp(B[0], own, rival, fit)
                np.testing.assert_allclose(lp, report['pair_margins'][0][rival], atol=1e-7)
                counts['outer_LP_probes'] += 1
                counts['outer_prediction_replays'] += 1
            counts['outer_count_matches'] += 2
        print(name, 'outer' if outer else 'inner'+str(index), 'original paths/fits/FRS/selection audited', flush=True)
    counts['datasets'] += 1
    counts['outer_folds'] += 1
    counts['inner_folds'] += len(saved['inner'])
    sizes = {m: len(saved['outer']['results'][m]['features']) for m in METHODS}
    counts['selected_full_proposed'] += sizes['proposed'] == saved['d']
    counts['empty_native_outputs'] += sizes['fgs_native'] == 0
    counts['empty_native_outputs'] += sizes['sfrip_native'] == 0
    row = {'dataset': name, 'metrics': outer_metrics, 'actual_counts': sizes, 'd': saved['d'],
           'removed': 1-sizes['proposed']/saved['d']}
    result = {'row': row, 'counts': dict(counts)}
    checkpoint.write_text(json.dumps({'fingerprints': fp, 'result': result}, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    names = [r['dataset'] for r in json.loads((ROOT/'results/development-sample-depth-audit.json').read_text())['rows']]
    if args.partial:
        names = [name for name in names if (FOLDER/(name+'.json')).exists() and json.loads((FOLDER/(name+'.json')).read_text())['complete']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        results = [f.result() for f in as_completed([pool.submit(audit, name) for name in names])]
    if args.partial:
        print('partial audit complete', names, flush=True)
    else:
        saved = json.loads((FOLDER/'complete.json').read_text())
        lookup = {r['row']['dataset']: r for r in results}
        counts = sum((Counter(r['counts']) for r in results), Counter())
        for k, v in saved['counts'].items():
            assert counts[k] == v
        for row in saved['rows']:
            equivalent(row, lookup[row['dataset']]['row'])
        rows = [lookup[name]['row'] for name in names]
        means = {m: {k: float(np.mean([r['metrics'][m][k] for r in rows])) for k in METRICS} for m in METHODS}
        equivalent(means, saved['means'])
        contrasts = {m: paired([r['metrics']['proposed']['clean_BA']-r['metrics'][m]['clean_BA'] for r in rows])
                     for m in METHODS if m != 'proposed'}
        equivalent(contrasts, saved['contrasts_clean_BA'])
        primary = ('fgs_matched', 'sfrip_matched')
        equivalent(holm({m: contrasts[m]['sign_flip_two_sided_p'] for m in primary}), saved['primary_descriptive_holm'])
        removed = float(np.mean([r['removed'] for r in rows]))
        equivalent(removed, saved['mean_removed'])
        gate = all(contrasts[m]['mean_pp'] >= .5 and contrasts[m]['wins'] >= 10 for m in primary)
        gate = gate and contrasts['full']['mean_pp'] >= 0 and removed >= .1
        gate = gate and counts['teacher_optimizer_caps'] == counts['downstream_optimizer_caps'] == 0
        assert gate == saved['pilot_gate_passed'] and not saved['goal_achieved']
        report = {'passed': True, 'counts': dict(counts), 'pilot_gate_passed': bool(gate),
                  'artifact_hashes': {name: digest(FOLDER/(name+'.json')) for name in names},
                  'auditor_sha256': digest(Path(__file__)), 'summary_sha256': digest(FOLDER/'complete.json'),
                  'goal_achieved': False, 'scope': 'all_original_baseline_paths_independent_classifier_design_FRS_and_selection_development'}
        (FOLDER/'complete-audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(report, flush=True)
