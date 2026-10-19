"""Versioned extension; never mutate any frozen source or scientific output."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def write_new(relative, text):
    path = ROOT / relative
    assert not path.exists(), path
    path.write_text(text, encoding='utf-8')


def main():
    source = ROOT/'scripts/fit_pair_capped_frs_hinge.py'
    code = source.read_text(encoding='utf-8')
    code = replace_once(code, 'def fit_pair_capped_frs_target(', 'def fit_fixed_face_frs_target(')
    code = replace_once(code, "constrained=True, margin=.02,", "constrained=True, fixed_zero_indices=(), optimization_eps=.01, margin=.02,")
    code = replace_once(code, 'mass = float(np.sqrt(A.shape[1])); initial = anchor / mass',
        "zeros = list(fixed_zero_indices)\n    if (zeros != sorted(set(zeros)) or any(int(j) != j or not 0 <= j < len(fs) for j in zeros)\n        or optimization_eps not in (0., .01)):\n        raise ValueError('valid sorted fixed zeros and nominal/robust optimization radius required')\n    mass = float(np.sqrt(A.shape[1])); initial = anchor / mass")
    code = replace_once(code, 'initial, margin=margin)', 'initial, margin=margin, eps=optimization_eps)')
    code = replace_once(code, 'bounds = [(0, 1)]*p +', 'bounds = [(0., 0.) if j in zeros else (0., 1.) for j in range(p)] +')
    code = replace_once(code, "or not evaluate_class_cell(cell, requested_initial)['feasible']):",
        "or not np.all(requested_initial[zeros] == 0.)\n        or (len(C) and not np.all(C@requested_initial > rhs))\n        or not all(v < cell['eta'] for v in evaluate_class_cell(cell, requested_initial)['raw_upper_bounds_with_inflation'])):")
    code = replace_once(code, "use_requested = bool(initial_value <= start)\n    current_actual = requested_initial.copy() if use_requested else anchor.copy()",
        "# The original anchor need not belong to this face. Never fall back to it.\n    use_requested = True\n    current_actual = requested_initial.copy()")
    code = replace_once(code, 'actual, repair_trace, blend = repair_cell_metric(C, rhs, cell, candidate*mass, anchor)',
        "actual, repair_trace, blend = repair_cell_metric(C, rhs, cell, candidate*mass, requested_initial)\n            assert np.all(actual[zeros] == 0.), 'repair must preserve the entire fixed face'")
    code = replace_once(code, "'anchor_blend': blend", "'face_seed_blend': blend")
    code = replace_once(code, 'assert oracle.objective(current_actual/mass, reg=reg) <= start+1e-10',
        "assert oracle.objective(current_actual/mass, reg=reg) <= start_point_value+1e-10\n    assert np.all(current_actual[zeros] == 0.)")
    code = replace_once(code, "'margin': margin, 'reg': reg,", "'fixed_zero_indices':zeros, 'optimization_eps':optimization_eps, 'certificate_eps':cell['eps'],\n        'margin': margin, 'reg': reg,")
    code = replace_once(code, "'scope': 'bounded_error_surrogate_and_monotone_FRS_class_constraints_inside_finite_budget_DC_not_global_optimum_or_prediction_advantage'",
        "'scope': 'fixed_certified_face_prediction_loss_descent_relative_to_face_seed_original_reference_unchanged_not_accuracy_guarantee'")
    code = code.replace('Classwise FRS-constrained CCCP with replayable convex subproblems.',
        'Classwise FRS-constrained CCCP on an immutable certified feature face.')
    write_new('scripts/fit_fixed_face_frs.py', code)

    audit_source = ROOT/'scripts/audit_pair_capped_frs_fit.py'
    audit = audit_source.read_text(encoding='utf-8')
    audit = replace_once(audit, 'def check_pair_capped_fit(', 'def check_fixed_face_fit(')
    audit = replace_once(audit, "info['eps'] == .01", "info['eps'] == fit['optimization_eps'] and fit['optimization_eps'] in (0., .01)\n    assert fit['certificate_eps'] == fit['class_cell']['eps'] == .01")
    audit = replace_once(audit, "assert target.min() >= 0 and abs(target.sum()-mass) < 1e-8",
        "zeros = fit['fixed_zero_indices']\n    assert zeros == sorted(set(zeros)) and all(int(j) == j and 0 <= j < p for j in zeros)\n    assert np.all(target[zeros] == 0.)\n    assert target.min() >= 0 and abs(target.sum()-mass) < 1e-8")
    audit = replace_once(audit, "assert scalar_cell_evaluation(fit['class_cell'], requested)['feasible']",
        "assert np.all(requested[zeros] == 0.)\n    assert not len(C) or np.all(C@requested > rhs)\n    assert all(v < .1 for v in scalar_cell_evaluation(fit['class_cell'], requested)['raw_upper_bounds_with_inflation'])")
    audit = replace_once(audit, "accepted = bool(fit['requested_initial_objective'] <= fit['training_objective_initial'])",
        "accepted = True")
    audit = replace_once(audit, "expected_start = requested if accepted else anchor", 'expected_start = requested')
    audit = replace_once(audit, "assert np.max(matrix@x-b) < 1e-6 and abs(eq@x-1) < 1e-7",
        "upper[zeros] = 0.\n            assert np.max(matrix@x-b) < 1e-6 and abs(eq@x-1) < 1e-7\n            assert np.all(x[zeros] == 0.), 'fixed face must appear in the actual primal solution'")
    audit = replace_once(audit, "equivalent(objective(v), step['candidate_objective'])",
        "assert np.all(v[zeros] == 0.)\n            equivalent(objective(v), step['candidate_objective'])")
    audit = replace_once(audit, "equivalent(current.tolist(), target.tolist()); assert objective(target/mass) <= objective(initial)+1e-10",
        "equivalent(current.tolist(), target.tolist()); assert objective(target/mass) <= objective(requested/mass)+1e-10")
    write_new('scripts/audit_fixed_face_frs_fit.py', audit)
    files = [source, audit_source, ROOT/'scripts/fit_fixed_face_frs.py', ROOT/'scripts/audit_fixed_face_frs_fit.py']
    write_new('research/fixed-face-FRS-prototype-provenance.json', json.dumps({
        'purpose':'Freeze certified zeros while reoptimizing predictive weights; no theoretical originality or superiority claim.',
        'original_reference_preserved':True, 'face_seed_fallback_only':True,
        'certificate_eps':.01, 'eta':.1, 'retention':.9,
        'hashes':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    }, indent=2))


if __name__ == '__main__':
    main()
