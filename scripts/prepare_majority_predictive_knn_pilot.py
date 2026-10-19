"""Create a separate fair dual-target pilot; preserve every frozen ancestor."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new)


def save(name, text):
    path = ROOT/'scripts'/name
    if path.exists(): raise FileExistsError(path)
    path.write_text(text, encoding='utf-8')


def main():
    core = (ROOT/'scripts/majority_constrained_predictive_metric.py').read_text(encoding='utf-8')
    core = once(core, '    targets, witnesses, matrices = [], [], []', '''    original_ok = all(
        np.sum((y == c) & protected & (bound > 1e-10)) >= int(np.ceil(retention*np.sum((y == c) & protected)-1e-12))
        for c in np.unique(y))
    protection_reference = 'original_full' if original_ok else 'own_anchor'
    if not original_ok: protected = bound > 1e-10
    targets, witnesses, matrices = [], [], []''')
    core = once(core, "'retention': retention, 'eps': eps,", "'retention': retention, 'eps': eps, 'protection_reference': protection_reference,")
    core = once(core, '    _, validation = fit_predictive_target(A, y, features, anchor, temperature=temperature, reg=reg, max_iter=1)',
                '    _, validation = fit_predictive_target(A, y, features, anchor, temperature=temperature, reg=reg, max_iter=1)\n    validation = {**validation, \'max_iter\': max_iter}')
    save('adaptive_majority_predictive_metric.py', core)
    runner = (ROOT/'scripts/run_predictive_target_knn_pilot.py').read_text(encoding='utf-8')
    runner = runner.replace('from predictive_target_knn_stage import', 'from majority_predictive_knn_stage import')
    runner = runner.replace("ROOT/'results/predictive-target-knn-pilot'", "ROOT/'results/majority-predictive-knn-pilot'")
    runner = runner.replace('scripts/predictive_target_knn_stage.py', 'scripts/majority_predictive_knn_stage.py')
    runner = runner.replace('research/predictive-target-knn-pilot-protocol.md', 'research/majority-predictive-knn-pilot-protocol.md')
    runner = once(runner, "ROOT/'scripts/capped_predictive_metric.py',", "ROOT/'scripts/capped_predictive_metric.py', ROOT/'scripts/adaptive_majority_predictive_metric.py',")
    runner = once(runner, "ROOT/'results/predictive-target-metric-admission/complete-audit.json',", "ROOT/'results/predictive-target-metric-admission/complete-audit.json', ROOT/'results/majority-constrained-target-admission-v3/complete-audit.json',")
    runner = runner.replace('all15_exposed_development_certified_predictive_target_metric_nested_selection_not_confirmation',
                            'all15_development_dual_metric_families_equal_weighted_controls_not_confirmation')
    save('run_majority_predictive_knn_pilot.py', runner)
    audit = (ROOT/'scripts/audit_predictive_target_knn_pilot.py').read_text(encoding='utf-8')
    audit = audit.replace('from run_predictive_target_knn_pilot import', 'from run_majority_predictive_knn_pilot import')
    start, end = audit.index('GRID = '), audit.index('\ndef check_metric')
    audit = audit[:start]+'''from majority_predictive_knn_stage import GRID
from audit_predictive_target_knn_pilot import check_stage as old_check_stage
from audit_majority_constrained_target_admission import independent_polytope, independent_objective
from audit_flexible_majority_repair_admission import friend_enemy_bounds


def check_stage(A, y, block, counts):
    ordinary = [r for r in block['learners'] if r['family'] == 'unconstrained']
    old_check_stage(A, y, {'learners': ordinary, 'certificate_checks': block['certificate_checks']}, counts)
    d, mass = A.shape[1], np.sqrt(A.shape[1]); full = np.ones(d)/mass
    original = independent_protected(A, y, full); keys = set()
    for r in block['learners']:
        key = (tuple(r['features']), tuple(r['anchor']), r['family'])
        assert key not in keys; keys.add(key)
        if r['family'] == 'unconstrained': continue
        assert r['family'] == 'majority_constrained'
        fs, anchor, target, fit = r['features'], np.array(r['anchor']), np.array(r['target']), r['fit']
        for w in (anchor, target):
            assert w.min() >= 0 and w.max()/mass <= 2/d+1e-10
            np.testing.assert_allclose(w.sum(), mass, atol=1e-10)
        equivalent(fit['weights'], target.tolist())
        assert fit['temperature'] == .05 and fit['reg'] == .01 and fit['max_iter'] == 200
        if fit['polytope'] is None:
            assert fit['status'] in ('constrained_unique_capped_simplex', 'constrained_no_LOO_same_class_supervision')
            equivalent(anchor.tolist(), target.tolist())
        else:
            protected = original
            own = np.zeros(len(A), bool)
            for i in range(len(A)):
                values = friend_enemy_bounds(A[:, fs], y, anchor, i)
                if values is not None:
                    raw = np.partition(values, 2)[2]; own[i] = raw-1e-10*(1+abs(raw)) > 1e-10
            original_ok = all(sum((y == c) & original & own) >= int(np.ceil(.9*sum((y == c) & original)-1e-12)) for c in np.unique(y))
            expected = 'original_full' if original_ok else 'own_anchor'
            assert fit['polytope']['protection_reference'] == expected
            if not original_ok: protected = own
            C, rhs = independent_polytope(A[:, fs], y, anchor, protected, fit['polytope'])
            counts['independent_fixed_witness_rows'] += len(C)
            initial = independent_objective(A[:, fs], y, anchor/mass)
            equivalent(initial, fit['training_objective_initial'])
            if fit['status'] == 'local_majority_constrained_target':
                assert fit['optimizer_success'] and fit['linear_constraints_verified']
                assert not len(C) or np.all(C@target >= rhs)
                final = independent_objective(A[:, fs], y, target/mass)
                equivalent(final, fit['training_objective_final']); assert final <= initial+1e-8
            else:
                assert 'fallback' in fit['status']; equivalent(anchor.tolist(), target.tolist())
        counts['independent_target_objective_replays'] += 1
        counts['target_optimizer_failures'] += not fit['optimizer_success']

''' + audit[end:]
    audit = once(audit, "    assert metric['interpolation'] == GRID[target_index]\n    requested = (1-GRID[target_index])*anchor+GRID[target_index]*np.array(learner['target'])", "    family, t = GRID[target_index]\n    assert metric['interpolation'] == t and metric['target_family'] == family == learner['family']\n    requested = (1-t)*anchor+t*np.array(learner['target'])")
    save('audit_majority_predictive_knn_pilot.py', audit)


if __name__ == '__main__': main()
