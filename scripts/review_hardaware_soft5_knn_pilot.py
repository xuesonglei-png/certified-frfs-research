"""Complete audited cohort review; no query-based model or dataset selection."""
import json
from collections import Counter
from pathlib import Path

import numpy as np

from run_hardaware_soft5_knn_pilot import ROOT, FOLDER, PARENT, METHODS, digest


def verified_summary(folder):
    summary = json.loads((folder/'complete.json').read_text())
    audit = json.loads((folder/'complete-audit.json').read_text())
    assert audit['passed'] and audit['summary_sha256'] == digest(folder/'complete.json')
    assert len(summary['rows']) == 15
    assert set(audit['artifact_hashes']) == {r['dataset'] for r in summary['rows']}
    for name, sha in audit['artifact_hashes'].items():
        assert digest(folder/(name+'.json')) == sha
    return summary, audit


def main():
    summary, audit = verified_summary(FOLDER)
    parent, _ = verified_summary(PARENT)
    previous, _ = verified_summary(ROOT/'results/predictive-target-knn-pilot')
    previous_rows = {r['dataset']: r for r in previous['rows']}
    dual, _ = verified_summary(ROOT/'results/simplex-soft5-knn-pilot')
    dual_rows = {r['dataset']: r for r in dual['rows']}
    old_rows = {r['dataset']: r for r in parent['rows']}
    assert set(old_rows) == {r['dataset'] for r in summary['rows']}
    rows, counts = [], Counter()
    for row in sorted(summary['rows'], key=lambda r: r['dataset']):
        name = row['dataset']; saved = json.loads((FOLDER/(name+'.json')).read_text())
        assert saved['complete'] and not saved['query_labels_used_for_selection']
        outer = saved['outer']; result = outer['results']['proposed']; metric = result['metric']
        selected = saved['selection']['selected']['proposed']
        choice = saved['selection']['options']['proposed'][selected]
        lid = metric['learner_id']; learner = outer['learners'][lid]
        did = metric['deployed_guard_id']
        guards = {g['id']: g for g in outer['certificate_checks']}
        assert guards[did]['accepted']
        ba = {m: row['metrics'][m]['clean_BA'] for m in METHODS}
        parent_ba = old_rows[name]['metrics']['proposed']['clean_BA']
        inner_selected = [b['options']['proposed'][selected]['metric'] for b in saved['inner']]
        all_blocks = saved['inner']+[outer]
        for block in all_blocks:
            counts['learners'] += len(block['learners'])
            counts['guard_checks'] += len(block['certificate_checks'])
            counts['guard_rejections'] += sum(not c['accepted'] for c in block['certificate_checks'])
            counts['optimizer_failures'] += sum(not c['fit'].get('optimizer_success', True) for c in block['learners'])
        counts['outer_selected_guard_fallbacks'] += metric['guard_fallback']
        counts['inner_selected_guard_fallbacks'] += sum(m['guard_fallback'] for m in inner_selected)
        counts['outer_selected_full_supports'] += len(metric['features']) == saved['d']
        counts['outer_selected_nonzero_target_steps'] += metric['interpolation'] > 0
        deployed_shift = float(np.abs(np.array(metric['weights'])-learner['anchor']).sum())
        counts['outer_selected_actual_weight_changes'] += deployed_shift > 1e-8
        rows.append({
            'dataset': name, 'd': row['d'], 'actual_count': len(metric['features']),
            'removed': row['removed'], 'base_origin': metric['base_origin'],
            'interpolation': metric['interpolation'], 'target_family': metric['target_family'],
            'protection_reference': (learner['fit'].get('polytope') or {}).get('protection_reference'),
            'guard_reference': learner['fit'].get('guard_reference'),
            'selected_guard_references_by_control': {m: (outer['learners'][outer['results'][m]['metric']['learner_id']]['fit'].get('guard_reference') if outer['results'][m]['metric']['learner_id'] is not None else None) for m in METHODS},
            'actual_anchor_shift_l1': deployed_shift,
            'selected_guard_fallback': metric['guard_fallback'],
            'inner_selected_guard_fallbacks': sum(m['guard_fallback'] for m in inner_selected),
            'inner_selected_full_supports': sum(len(m['features']) == saved['d'] for m in inner_selected),
            'inner_selected_BA': choice['mean_clean_BA'], 'outer_BA': ba['proposed'],
            'inner_to_outer_gap_pp': 100*(ba['proposed']-choice['mean_clean_BA']),
            'gain_over_previous_fixed_metric_pp': 100*(ba['proposed']-parent_ba),
            'gain_over_previous_anisotropic_soft5_target_pp': 100*(ba['proposed']-dual_rows[name]['metrics']['proposed']['clean_BA']),
            'gain_over_previous_unconstrained_target_pp': 100*(ba['proposed']-previous_rows[name]['metrics']['proposed']['clean_BA']),
            'contrasts_pp': {m: 100*(ba['proposed']-ba[m]) for m in METHODS if m != 'proposed'},
            'mean_perturbed_BA': row['metrics']['proposed']['mean_perturbed_BA'],
            'query_certified_BA': row['metrics']['proposed']['query_certified_BA'],
            'target_fit_status': learner['fit']['status'],
            'target_objective_initial': learner['fit']['training_objective_initial'],
            'target_objective_final': learner['fit']['training_objective_final'],
        })
    checks = {
        'both_primary_mean_gains_at_least_half_pp': all(summary['contrasts_clean_BA'][m]['mean_pp'] >= .5 for m in ('fgs_matched', 'sfrip_matched')),
        'both_primary_wins_at_least_ten': all(summary['contrasts_clean_BA'][m]['wins'] >= 10 for m in ('fgs_matched', 'sfrip_matched')),
        'mean_removal_at_least_ten_percent': summary['mean_removed'] >= .1,
        'nonnegative_full_gap': summary['contrasts_clean_BA']['full']['mean_pp'] >= 0,
        'zero_optimizer_failures': counts['optimizer_failures'] == 0,
    }
    assert all(checks.values()) == summary['pilot_gate_passed']
    review = {
        'rows': rows, 'counts': dict(counts), 'pilot_gate_checks': checks,
        'pilot_gate_passed': summary['pilot_gate_passed'], 'goal_achieved': False,
        'mean_gain_over_previous_fixed_metric_pp': float(np.mean([r['gain_over_previous_fixed_metric_pp'] for r in rows])),
        'mean_gain_over_previous_anisotropic_soft5_target_pp': float(np.mean([r['gain_over_previous_anisotropic_soft5_target_pp'] for r in rows])),
        'mean_gain_over_previous_unconstrained_target_pp': float(np.mean([r['gain_over_previous_unconstrained_target_pp'] for r in rows])),
        'primary_contrasts': {m: summary['contrasts_clean_BA'][m] for m in ('fgs_matched', 'sfrip_matched')},
        'primary_descriptive_holm': summary['primary_descriptive_holm'],
        'negative_datasets_by_control': {m: [r['dataset'] for r in rows if r['contrasts_pp'][m] < -1e-9] for m in METHODS if m != 'proposed'},
        'limitations': [
            'All 15 datasets and reused folds are exposed development; p-values cannot confirm superiority.',
            'Target-objective reduction does not guarantee hard-5NN or population accuracy.',
            'Finite-reference guards and query certificates have separate scopes.',
            'All weighted controls received cap1, all three target families and the same 21 options, two source fits, twelve seed checks and four-iteration/sixteen-check local budgets.',
            'Native methods and count-matched adaptations answer different comparisons.',
            'Own-anchor guards used by infeasible controls are not original-full FRS preservation.',
            'Hard-LOO training improvements are not unbiased validation; local budget stops do not prove stationarity.',
            'FD+/FCE+ faithful reproduction and fresh confirmation remain unresolved.',
        ],
        'summary_sha256': digest(FOLDER/'complete.json'),
        'audit_sha256': digest(FOLDER/'complete-audit.json'),
        'review_script_sha256': digest(Path(__file__)),
    }
    target = ROOT/'research/hardaware-soft5-knn-pilot-review.json'
    target.write_text(json.dumps(review, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in review.items() if k != 'rows'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
