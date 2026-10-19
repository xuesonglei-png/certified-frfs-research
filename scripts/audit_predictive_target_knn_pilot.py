"""Independent targets/objectives, guard replay,5NN and fold selection audit."""
import json
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_predictive_target_knn_pilot import ROOT, FOLDER, PARENT, METHODS, COUNTS, WEIGHTED, digest
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.feasible_transport import weight_certificate
from audit_flexible_majority_repair_admission import independent_protected
from audit_classwise_repair_admission import independent_class_certificate
from audit_predictive_target_metric_admission import objective
from audit_certified_metric_knn_pool import independent_evaluate, query_certificate
from audit_multiscale_certified_metric_knn_pool import policies
from audit_forest_compact_transfer import equivalent
from review_box_iterations import paired
from run_nested_joint_path import holm

GRID = (0., .05, .1, .25, .5, .75, 1.)


def check_stage(A, y, block, counts):
    d, mass = A.shape[1], np.sqrt(A.shape[1]); full = np.ones(d)/mass
    protected = independent_protected(A, y, full)
    keys = []
    for learner in block['learners']:
        fs, anchor, target = learner['features'], np.array(learner['anchor']), np.array(learner['target'])
        key = (tuple(fs), tuple(anchor)); assert key not in keys; keys.append(key)
        assert fs == sorted(set(fs)) and min(fs) >= 0 and max(fs) < d
        for weights in (anchor, target):
            assert weights.shape == (len(fs),) and weights.min() >= 0 and weights.max()/mass <= 2/d+1e-10
            np.testing.assert_allclose(weights.sum(), mass, atol=1e-10)
        fit = learner['fit']; equivalent(fit['weights'], target.tolist())
        assert fit['features'] == fs and fit['temperature'] == .05 and fit['reg'] == .01 and fit['max_iter'] == 200
        equivalent(fit['mass'], mass); equivalent(fit['cap_normalized'], 2/d)
        labels, sizes = np.unique(y, return_counts=True)
        assert fit['singleton_training_classes'] == labels[sizes == 1].tolist()
        if fit['status'] == 'unique_capped_simplex':
            assert abs(len(fs)*2/d-1) <= 1e-12 and fit['optimizer_iterations'] == 0
            np.testing.assert_allclose(target, np.full(len(fs), mass/len(fs)), atol=1e-10)
        elif fit['status'] == 'no_LOO_same_class_supervision':
            assert np.all(sizes == 1); equivalent(anchor.tolist(), target.tolist())
        else:
            start = (1-1e-5)*anchor/mass+1e-5*np.ones(len(fs))/len(fs)
            initial = objective(A[:, fs], y, start); equivalent(initial, fit['training_objective_initial'])
            if fit['status'] == 'local_predictive_target':
                assert fit['optimizer_success'] and fit['optimizer_status'] == 0
                final = objective(A[:, fs], y, target/mass); equivalent(final, fit['training_objective_final'])
                assert final <= initial+1e-8
            else:
                assert 'fallback' in fit['status']; equivalent(anchor.tolist(), target.tolist())
        counts['independent_target_objective_replays'] += 1
        counts['target_optimizer_failures'] += not fit['optimizer_success']
    checks = block['certificate_checks']; seen = set()
    for i, check in enumerate(checks):
        assert check['id'] == i; fs, w = check['features'], np.array(check['weights'])
        key = (tuple(fs), tuple(w)); assert key not in seen; seen.add(key)
        fresh = weight_certificate(A, y, fs, full, w, protected=protected)
        classes = independent_class_certificate(y, fresh)
        equivalent(check['certificate'], fresh); equivalent(check['class_certificate'], classes)
        assert check['accepted'] == (fresh['feasible'] and classes['feasible_all_classes'])
        counts['independent_guard_replays'] += 1; counts['guard_rejections'] += not check['accepted']


def check_metric(metric, base, block, method, target_index, counts):
    fs, anchor = base['features'], np.array(base['weights'])
    assert metric['features'] == fs and metric['base_origin'] == base['origin'] and metric['target_index'] == target_index
    if method not in WEIGHTED:
        assert target_index == 0 and metric['learner_id'] is metric['guard_id'] is metric['deployed_guard_id'] is None
        assert not metric['guard_fallback'] and metric['interpolation'] == 0.
        equivalent(metric['weights'], anchor.tolist()); return
    learner = block['learners'][metric['learner_id']]
    assert learner['features'] == fs; equivalent(learner['anchor'], anchor.tolist())
    assert metric['interpolation'] == GRID[target_index]
    requested = (1-GRID[target_index])*anchor+GRID[target_index]*np.array(learner['target'])
    equivalent(requested.tolist(), metric['requested_weights'])
    if method == 'proposed':
        check = block['certificate_checks'][metric['guard_id']]
        assert check['features'] == fs; equivalent(check['weights'], requested.tolist())
        fallback = not check['accepted']; assert fallback == metric['guard_fallback']
        deployed = block['certificate_checks'][metric['deployed_guard_id']]
        assert deployed['accepted'] and deployed['features'] == fs
        final = anchor if fallback else requested
        equivalent(deployed['weights'], final.tolist()); equivalent(metric['weights'], final.tolist())
        counts['proposed_fallback_options'] += fallback
    else:
        assert metric['guard_id'] is metric['deployed_guard_id'] is None and not metric['guard_fallback']
        equivalent(metric['weights'], requested.tolist())


def independent_choice(rows):
    for row in rows:
        for k in ('clean_BA', 'balanced_logloss', 'actual_count'):
            equivalent(row['mean_'+k], float(np.mean([r[k] for r in row['folds']])))
    return min(range(len(rows)), key=lambda i: (-round(rows[i]['mean_clean_BA'], 12), round(rows[i]['mean_balanced_logloss'], 12),
        rows[i]['mean_actual_count'], rows[i]['selector_index'], rows[i]['target_index']))


def audit(name):
    threadpool_limits(1)
    artifact = FOLDER/(name+'.json'); saved = json.loads(artifact.read_text())
    assert saved['complete'] and not saved['query_labels_used_for_selection']
    for rel, sha in saved['fingerprints'].items(): assert digest(ROOT/rel) == sha, rel
    parent_path = PARENT/(name+'.json'); parent = json.loads(parent_path.read_text()); parent_audit = json.loads((PARENT/'complete-audit.json').read_text())
    assert parent_audit['passed'] and parent_audit['artifact_hashes'][name] == digest(parent_path)
    assert saved['reference_indices'] == parent['reference_indices'] and saved['validation_indices'] == parent['validation_indices']
    X, y = load_dataset(name); ri, vi = np.array(saved['reference_indices']), np.array(saved['validation_indices'])
    assert not set(ri)&set(vi); raw, labels = X[ri], y[ri]
    counts = Counter(datasets=1); combined, selected = saved['selection']['options'], saved['selection']['selected']
    for m in METHODS:
        width = len(GRID) if m in WEIGHTED else 1
        assert len(combined[m]) == COUNTS[m]*width
        for i, row in enumerate(combined[m]):
            assert row['selector_index'] == i//width and row['target_index'] == i%width
            equivalent(row['folds'], [b['options'][m][i]['folds'][0] for b in saved['inner']])
        assert selected[m] == independent_choice(combined[m])
    recipe_index = combined['proposed'][selected['proposed']]['selector_index']
    blocks = parent['candidate_reference_blocks']; assert len(saved['inner']) == len(blocks)-1
    for source, block in zip(blocks[:-1], saved['inner']):
        assert block['reference_indices'] == source['reference_indices'] and block['validation_indices'] == source['validation_indices']
        ca, cb = np.array(block['reference_indices']), np.array(block['validation_indices']); assert not set(ca)&set(cb)
        A, B = preprocess_fit(raw[ca], raw[cb]); check_stage(A, labels[ca], block, counts)
        count = len(policies(A, source)['proposed'][recipe_index]['features']); assert count == block['matched_proposed_count']
        bases = policies(A, source, count=count); prediction_inputs = {}
        for m in METHODS:
            width = len(GRID) if m in WEIGHTED else 1; rows = block['options'][m]
            assert len(rows) == COUNTS[m]*width
            for i, row in enumerate(rows):
                si, ti = i//width, i%width
                assert row['selector_index'] == si and row['target_index'] == ti
                check_metric(row['metric'], bases[m][si], block, m, ti, counts)
                mid = row['prediction_id']; key = (tuple(row['metric']['features']), tuple(row['metric']['weights']))
                if mid in prediction_inputs: assert prediction_inputs[mid] == key
                prediction_inputs[mid] = key
                report = block['predictions'][mid]
                equivalent(row['folds'], [{**report['metrics'], 'actual_count': len(key[0])}]); counts['inner_option_replays'] += 1
        assert sorted(prediction_inputs) == list(range(len(block['predictions'])))
        for mid, (fs, w) in prediction_inputs.items():
            fresh = independent_evaluate(A, labels[ca], B, labels[cb], list(fs), w)
            equivalent(fresh, block['predictions'][mid]); counts['independent_inner_prediction_replays'] += 1
        counts['inner_folds'] += 1
        print(name, 'inner', counts['inner_folds'], 'target/guard/predictions audited', flush=True)
    A, B = preprocess_fit(raw, X[vi]); block = saved['outer']; check_stage(A, labels, block, counts)
    count = len(policies(A, blocks[-1])['proposed'][recipe_index]['features']); assert count == block['matched_proposed_count']
    bases = policies(A, blocks[-1], count=count); metrics, sizes = {}, {}
    for m in METHODS:
        result = block['results'][m]; row = combined[m][selected[m]]
        assert result['option_index'] == selected[m] and result['selector_index'] == row['selector_index'] and result['target_index'] == row['target_index']
        metric = result['metric']; check_metric(metric, bases[m][row['selector_index']], block, m, row['target_index'], counts)
        fs, w = metric['features'], metric['weights']; report = independent_evaluate(A, labels, B, y[vi], fs, w)
        cert = query_certificate(A, labels, B, np.array(report['predictions']), fs, w); equivalent(cert, result['query_certificate'])
        correct = (np.array(report['predictions']) == y[vi]) & np.array(cert['certified'])
        report['metrics']['query_certified_BA'] = float(np.mean([np.mean(correct[y[vi] == c]) for c in np.unique(y[vi])]))
        noise = []; assert len(result['perturbation_evaluations']) == 16
        for draw, stored in enumerate(result['perturbation_evaluations']):
            rng = np.random.default_rng(20261060+draw)
            noisy_A = np.clip(A+rng.uniform(-.01, .01, A.shape), 0, 1); noisy_B = np.clip(B+rng.uniform(-.01, .01, B.shape), 0, 1)
            fresh = independent_evaluate(noisy_A, labels, noisy_B, y[vi], fs, w); equivalent(fresh, stored)
            assert not np.any((np.array(fresh['predictions']) != report['predictions']) & np.array(cert['certified']))
            noise.append(fresh['metrics']['clean_BA']); counts['shared_perturbation_predictions'] += 1
        report['metrics']['mean_perturbed_BA'] = float(np.mean(noise)); equivalent(report, result['evaluation'])
        metrics[m] = report['metrics']; sizes[m] = len(fs); counts['outer_predictions'] += 1
    assert count == sizes['proposed'] == sizes['fgs_matched'] == sizes['sfrip_matched'] == sizes['fgs_uniform_matched'] == sizes['sfrip_uniform_matched']
    row = {'dataset': name, 'd': saved['d'], 'actual_counts': sizes, 'removed': 1-count/saved['d'], 'metrics': metrics}
    print(name, 'COMPLETE independently audited', flush=True)
    return {'row': row, 'counts': dict(counts), 'artifact_sha256': digest(artifact)}


if __name__ == '__main__':
    summary = json.loads((FOLDER/'complete.json').read_text()); reports, counts = [], Counter()
    with ProcessPoolExecutor(max_workers=2) as ex:
        for future in as_completed([ex.submit(audit, r['dataset']) for r in summary['rows']]):
            report = future.result(); reports.append(report); counts.update(report['counts'])
    rows = sorted([r['row'] for r in reports], key=lambda r: r['dataset']); equivalent(rows, sorted(summary['rows'], key=lambda r: r['dataset']))
    means = {m: {k: float(np.mean([r['metrics'][m][k] for r in rows])) for k in rows[0]['metrics']['proposed']} for m in METHODS}
    equivalent(means, summary['means'])
    contrasts = {m: paired([r['metrics']['proposed']['clean_BA']-r['metrics'][m]['clean_BA'] for r in rows]) for m in METHODS if m != 'proposed'}
    equivalent(contrasts, summary['contrasts_clean_BA']); primary = ('fgs_matched', 'sfrip_matched')
    equivalent(holm({m: contrasts[m]['sign_flip_two_sided_p'] for m in primary}), summary['primary_descriptive_holm'])
    removed = float(np.mean([r['removed'] for r in rows])); equivalent(removed, summary['mean_removed'])
    for audited_key, production_key in (('independent_target_objective_replays', 'target_fits'), ('independent_guard_replays', 'guard_checks'),
                                       ('guard_rejections', 'guard_rejections'), ('target_optimizer_failures', 'target_optimizer_failures'), ('inner_folds', 'inner_folds'), ('datasets', 'datasets')):
        assert counts[audited_key] == summary['counts'][production_key]
    assert counts['datasets'] == 15 and counts['inner_folds'] == 44
    gate = all(contrasts[m]['mean_pp'] >= .5 and contrasts[m]['wins'] >= 10 for m in primary) and removed >= .1 and contrasts['full']['mean_pp'] >= 0 and counts['target_optimizer_failures'] == 0
    assert gate == summary['pilot_gate_passed'] and not summary['goal_achieved']
    output = {'passed': True, 'counts': dict(counts), 'artifact_hashes': {r['row']['dataset']: r['artifact_sha256'] for r in reports},
              'summary_sha256': digest(FOLDER/'complete.json'), 'auditor_sha256': digest(Path(__file__)), 'goal_achieved': False,
              'scope': 'independent_targets_objectives_mass_caps_original_classwise_guard_nested_scalar5NN_and_sharedquery_audit_not_confirmation'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(output, indent=2), encoding='utf-8'); print(output, flush=True)
