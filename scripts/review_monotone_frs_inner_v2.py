"""Review all development rows, equal controls, admission and search limits."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
from run_monotone_frs_inner import ROOT, FOLDER, METHODS, digest, frozen


def main():
    frozen(); summary = json.loads((FOLDER/'complete.json').read_text())
    audit = json.loads((FOLDER/'complete-audit.json').read_text())
    assert audit['passed'] and audit['summary_sha256'] == digest(FOLDER/'complete.json')
    counts = Counter(); trajectories = []; choices = Counter()
    for row in summary['rows']:
        name = row['dataset']; path = FOLDER/(name+'.json')
        assert audit['artifact_hashes'][name] == digest(path)
        saved = json.loads(path.read_text())
        for m in METHODS:
            ti = saved['selection'][m]['selected']; choices[(m, 'new_cell' if ti >= 21 else 'previous')] += 1
        for fi, block in enumerate(saved['inner']):
            for learner in block['stage']['learners']:
                if learner['family'] != 'one_sided_cell_constrained':
                    continue
                fit = learner['fit']; label = 'proposed' if learner['certified_arm'] else 'controls'
                counts[f'{label}_fits'] += 1; counts[f'{label}_unavailable_cells'] += fit['cell_unavailable']
                counts[f'{label}_own_anchor_references'] += fit['reference_scope'] == 'own_anchor'
                if fit['cell_unavailable']:
                    continue
                modes = Counter(r['mode'] for r in fit['class_cell']['classes'])
                counts[f'{label}_star_classes'] += modes['shared_star']
                counts[f'{label}_Lipschitz_classes'] += modes['signed_Lipschitz']
                gain = fit['training_objective_initial']-fit['training_objective_final']
                counts[f'{label}_training_strict_descent_fits'] += gain > 1e-10
                counts[f'{label}_optimizer_failures'] += fit['subproblem_failures']
                raw = fit['class_cell_final']['raw_upper_bounds_with_inflation']
                counts[f'{label}_active_class_bounds'] += sum(abs(v-.1) < 1e-7 for v in raw)
                counts[f'{label}_class_bounds'] += len(raw)
                displacement = float(np.sum(abs(np.asarray(learner['target'])-np.asarray(learner['anchor'])))/fit['mass'])
                trajectories.append({'dataset': name, 'fold': fi, 'arm_kind': label, 'features': learner['features'],
                    'reference_scope': fit['reference_scope'], 'training_objective_decrease': gain,
                    'normalized_weight_L1_change': displacement, 'modes': dict(modes),
                    'max_class_bound': max(raw), 'status': fit['status'], 'subproblem_failures': fit['subproblem_failures']})
            guards = block['stage']['monotone_certificate_checks']
            counts['new_guard_options'] += len(guards)
            counts['new_guard_admissions'] += sum(g['accepted'] and not g['old_accepted'] for g in guards)
            counts['new_guard_rejections'] += sum(not g['accepted'] for g in guards)
    out = {'mean_anchor_gain_pp': summary['mean_anchor_gain_pp'],
        'mean_vs_previous_DC_pp': summary['mean_vs_previous_DC_pp'],
        'mean_proposed_minus_controls_pp': summary['mean_proposed_minus_controls_pp'],
        'improved_vs_previous_DC_datasets': summary['improved_vs_previous_DC_datasets'],
        'target_optimizer_failures': summary['target_optimizer_failures'], 'pilot_gate_passed': summary['pilot_gate_passed'],
        'audit_sha256': digest(FOLDER/'complete-audit.json'), 'all_dataset_rows': summary['rows'],
        'mechanism_counts': dict(counts), 'selected_new_family_dataset_counts': {m: int(choices[(m, 'new_cell')]) for m in METHODS},
        'trajectories': trajectories, 'goal_achieved': False, 'classification_superiority_established': False,
        'new_outer_evaluation_performed': False,
        'decision': 'Fresh joint nested full-grid benchmark is now justified as development.' if summary['pilot_gate_passed'] else
            'Do not promote. Diagnose support restriction and active Lipschitz/stability bounds before a joint support-metric redesign.',
        'limits': ['All15 are repeatedly exposed development data; no confirmed significance claim.',
                   'Sequential frozen-support inner reselection is not fresh joint nested validation.',
                   'Same finite-table FRS guarantees do not imply test accuracy or population robustness.',
                   'Generic CCCP, order statistics, support functions and Lipschitz bounds are not themselves novel.']}
    (ROOT/'research/monotone-frs-inner-review.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k not in ('trajectories', 'all_dataset_rows')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
