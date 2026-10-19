"""Keep all development losses, active constraints, sparsity and strong controls."""
import json
from collections import Counter
from run_pair_transition_frs_inner import ROOT, FOLDER, METHODS, STRONG, digest, frozen
from continue_soft_topk_knn_pilot import verify


def main():
    frozen(); audit = verify(FOLDER); summary = json.loads((FOLDER/'complete.json').read_text())
    counts = Counter(); fits = []; selected_new = Counter()
    for row in summary['rows']:
        name = row['dataset']; path = FOLDER/(name+'.json')
        assert digest(path) == audit['artifact_hashes'][name]
        saved = json.loads(path.read_text())
        for m in METHODS:
            selected_new[m] += saved['selection'][m]['selected'] >= 28
        for fi, b in enumerate(saved['inner']):
            for learner in b['stage']['learners'][b['previous_audited_learners']:]:
                fit = learner['fit']; label = 'proposed' if learner['certified_arm'] else 'controls'
                counts[label+'_learner_records'] += 1
                counts[label+'_unavailable_cells'] += fit['cell_unavailable']
                counts[label+'_own_anchor_references'] += fit['reference_scope'] == 'own_anchor'
                zeros = sum(w == 0 for w in learner['target'])
                counts[label+'_exact_zero_weights'] += zeros
                counts[label+'_fits_with_exact_zero_weights'] += zeros > 0
                if fit['cell_unavailable']:
                    continue
                raw = fit['class_cell_final']['raw_upper_bounds_with_inflation']
                counts[label+'_active_class_bounds'] += sum(abs(v-.1) < 1e-7 for v in raw)
                counts[label+'_class_bounds'] += len(raw)
                counts[label+'_optimizer_failures'] += fit['subproblem_failures']
                counts[label+'_accepted_major_steps'] += sum(s['accepted_descent'] for s in fit['steps'])
                fits.append({'dataset':name, 'fold':fi, 'arm_kind':label,
                    'features':learner['features'], 'reference_scope':fit['reference_scope'],
                    'exact_zero_weights':zeros, 'max_class_bound':max(raw),
                    'training_objective_decrease':fit['training_objective_initial']-fit['training_objective_final'],
                    'clipped_training_bound_decrease_pp':100*(fit['training_error_upper_bound_initial']-fit['training_error_upper_bound_final']),
                    'pair_enemy_rows':sum(len(r['enemies']) for r in fit['class_cell']['pair_transition_objects']),
                    'subproblem_failures':fit['subproblem_failures'], 'status':fit['status']})
            guards = b['stage']['pair_certificate_checks']
            counts['pair_guard_records'] += len(guards)
            counts['additional_pair_guard_admissions'] += sum(g['accepted'] and not g['old_accepted'] for g in guards)
            counts['pair_guard_rejections'] += sum(not g['accepted'] for g in guards)
    review = {'audit_sha256':digest(FOLDER/'complete-audit.json'), 'all_dataset_rows':summary['rows'],
        'mean_vs_previous_DC_pp':summary['mean_vs_previous_DC_pp'],
        'mean_proposed_minus_controls_pp':summary['mean_proposed_minus_controls_pp'],
        'pilot_gate_passed':summary['pilot_gate_passed'], 'mechanism_counts':dict(counts),
        'selected_new_pair_family_dataset_counts':{m:selected_new[m] for m in METHODS},
        'trajectories':fits, 'goal_achieved':False, 'classification_superiority_established':False,
        'new_outer_evaluation_performed':False,
        'decision':'Fresh joint nested development is justified; no confirmation or submission claim.' if summary['pilot_gate_passed'] else
            'Do not promote to outer scoring. Review frozen support, active constraints and objective alignment.',
        'limits':summary['limits']+['Pair envelopes and CCCP have prior art; novelty unestablished.',
            'Full recent-baseline fidelity and unseen postfreeze confirmation remain required.']}
    (ROOT/'research/pair-transition-FRS-inner-review.json').write_text(json.dumps(review, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in review.items() if k not in ('all_dataset_rows', 'trajectories')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
