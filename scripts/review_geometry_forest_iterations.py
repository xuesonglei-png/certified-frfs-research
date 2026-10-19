"""Review all datasets without selecting favorable outcomes or editing TeX."""
import json
from collections import Counter
from review_box_iterations import ROOT


def main():
    native_dir = ROOT/'results/nested-geometry-forest'
    matched_dir = ROOT/'results/geometry-forest-matched-count'
    risk_dir = ROOT/'results/crossfit-forest-risk-pilot'
    native = json.loads((native_dir/'complete.json').read_text())
    matched = json.loads((matched_dir/'complete.json').read_text())
    risk = json.loads((risk_dir/'complete.json').read_text())
    group_dir = ROOT/'results/grouped-forest-ranking-pilot'
    learned_dir = ROOT/'results/learned-reference-geometry-pilot'
    group = json.loads((group_dir/'complete.json').read_text())
    learned = json.loads((learned_dir/'complete.json').read_text())
    joint_dir = ROOT/'results/joint-deletion-geometry-pilot'
    joint = json.loads((joint_dir/'complete.json').read_text())
    for folder in (native_dir, matched_dir, risk_dir, group_dir, learned_dir, joint_dir): assert json.loads((folder/'complete-audit.json').read_text())['passed']
    assert [r['dataset'] for r in native['rows']] == [r['dataset'] for r in matched['rows']] == [r['dataset'] for r in risk['rows']]
    counts = Counter(); caps = Counter(); zero_protected = []; selections = Counter(); failures = Counter()
    for row in native['rows']:
        name = row['dataset']; saved = json.loads((native_dir/(name+'.json')).read_text())
        for i, s in enumerate(saved['splits']):
            proposed = s['results']['proposed']; chosen = s['trace']['proposed'][proposed['chosen_index']]
            counts['outer_full_supports'] += len(proposed['features']) == saved['d']
            counts['outer_compact_supports'] += len(proposed['features']) < saved['d']
            selections[str((chosen['model_index'], chosen['fraction']))] += 1
            for c, n in s['stages']['proposed']['original_majority_protected_counts'].items():
                if n == 0: zero_protected.append({'dataset': name, 'outer_index': i, 'class': c})
            for method in ('fgs2023', 'sfrip2025'): counts[method+'_empty_outer'] += not s['results'][method]['features']
            for block in s['inner']+[s]:
                for fit in block['models']: caps[fit['scope']] += not fit['optimizer_converged']
                stages = block['stages']; stages = list(stages.values()) if isinstance(stages, dict) else stages
                for stage in stages:
                    for attempt in stage['path']['attempts']:
                        cert = block['geometry'][attempt['geometry_id']]['certificate']; counts['attempts'] += 1
                        if not cert['feasible']:
                            counts['rejected_attempts'] += 1
                            counts['majority_failure'] += not cert['majority_feasible']
                            counts['dependency_failure'] += cert['dependency_certificate']['upper_bound'] > .1
        pilot = json.loads((risk_dir/(name+'.json')).read_text())
        for record in pilot['records']:
            failures['same_selected_support_as_old'] += record['fits']['old']['features'] == record['fits']['crossfit']['features']
            failures['same_selected_count_as_old'] += len(record['fits']['old']['features']) == len(record['fits']['crossfit']['features'])
            failures['new_compact'] += len(record['fits']['crossfit']['features']) < pilot['d']
            for attempt in record['stage']['path']['attempts']:
                cert = attempt['certificate']
                failures['attempts'] += 1
                if not cert['feasible']:
                    failures['rejected'] += 1
                    failures['majority_failure'] += not cert['majority_feasible']
                    failures['dependency_failure'] += cert['dependency_certificate']['upper_bound'] > .1
    result = {'native_gate_passed': native['gate_passed'], 'risk_gate_passed': risk['gate_passed'],
        'native_counts': dict(counts), 'native_selections': dict(selections), 'optimizer_caps_by_scope': dict(caps),
        'zero_original_protected_classes': zero_protected, 'risk_support_diagnostics': dict(failures),
        'native_contrasts': native['clean_contrasts'], 'native_descriptive_holm_p': native['descriptive_holm_p'],
        'conditional_matched_count_contrasts': matched['contrasts'], 'risk_contrasts': risk['contrasts'],
        'group_pilot': {k:v for k,v in group.items() if k != 'rows'},
        'learned_reference_pilot': {k:v for k,v in learned.items() if k != 'rows'},
        'joint_deletion_pilot': {k:v for k,v in joint.items() if k != 'rows'},
        'goal_achieved': False,
        'limits': 'All datasets exposed development; FD+/FCE+ unresolved; native convention gaps; conditional matched budgets; no original classifier guard retained.'}
    (ROOT/'results/geometry-forest-iterations-review.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    def comparison(title, contrasts):
        lines = ['\n'+title, '', '|Comparator|Gain (pp)|Descriptive 95% interval (pp)|Sign-flip p|W/L/T|', '|---|---:|---|---:|---|']
        for name, v in contrasts.items():
            lines.append(f"|{name}|{v['mean_pp']:.4f}|[{v['bootstrap95_pp'][0]:.4f}, {v['bootstrap95_pp'][1]:.4f}]|{v['sign_flip_two_sided_p']:.6f}|{v['wins']}/{v['losses']}/{v['ties']}|")
        return lines
    lines = ['# Geometry-only fuzzy forest method iteration review', '',
        '2026-10-10. All six experiments and independent audits completed. No TeX edits or compilation.', '',
        '**Goal remains unachieved.** These exposed development results cannot establish final SOTA superiority. FRS support certification remains valid; original classifier-teacher margins are explicitly not guaranteed.', '',
        f"Native gate: {native['gate_passed']}; crossfit-risk pilot gate: {risk['gate_passed']}.", '',
        '|Dataset|Proposed BA %|Removed %|Full BA %|FGS native %|SFRIP native %|FGS same count %|SFRIP same count %|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row, control in zip(native['rows'], matched['rows']):
        m, c = row['methods'], control['methods']
        values = [m['proposed']['clean'], m['proposed']['removed'], m['full']['clean'], m['fgs2023']['clean'], m['sfrip2025']['clean'], c['fgs2023']['clean'], c['sfrip2025']['clean']]
        lines.append('|'+row['dataset']+'|'+'|'.join(f'{100*x:.3f}' for x in values)+'|')
    lines += comparison('## Native stopping controls: proposed minus comparator', native['clean_contrasts'])
    lines += ['', 'Holm adjusted descriptive p: '+str(native['descriptive_holm_p'])]
    lines += comparison('## Conditional exact-count adaptations: proposed minus comparator', matched['contrasts']['clean'])
    lines += ['', 'Counts are anchored to the proposed inner-selected policy; this asymmetric conditional diagnostic is not the original native baseline algorithm.']
    lines += comparison('## Fixed C1/penalty1/.8 pilot: new variant minus old projection-MSE path', risk['contrasts']['clean'])
    lines += ['', f"Old/new removal: {100*risk['means']['old']['removed']:.3f}% / {100*risk['means']['crossfit']['removed']:.3f}%. Matched diagnostic count mismatches: {risk['matched_count_mismatches']}/128.", '',
        '## Group-shrunk projection ranking pilot', '',
        f"Gain over old fixed policy: {group['contrasts']['clean']['grouped']['mean_pp']:.4f}pp; descriptive p={group['contrasts']['clean']['grouped']['sign_flip_two_sided_p']:.6f}. Gate={group['gate_passed']}. Selector fallbacks={group['counts']['selector_fallbacks']}/128. Actual removal={100*group['means']['grouped']['removed']:.3f}%. Independent objective/KKT verification is numerical, not a certified optimum.", '',
        '## Changed learned reference relation pilot', '',
        f"Gain over old fixed policy: {learned['contrasts']['clean']['learned']['mean_pp']:.4f}pp; descriptive p={learned['contrasts']['clean']['learned']['sign_flip_two_sided_p']:.6f}. Gate={learned['gate_passed']}. Removal old/new={100*learned['means']['old']['removed']:.3f}%/{100*learned['means']['learned']['removed']:.3f}%; count mismatches={learned['counts']['matched_count_mismatches']}/128.", '',
        'This pilot changes the protected full fuzzy relation. It preserves the learned relation, not the previous uniform relation. General weighted FRS thresholds are unchanged; this is not evidence that uniform-reference guarantees improved.', '',
        '## Same-reference certified joint-deletion pilot', '',
        f"Fixed fraction.5 gain over old identical fraction.5: {joint['contrasts']['clean']['joint']['mean_pp']:.4f}pp; descriptive p={joint['contrasts']['clean']['joint']['sign_flip_two_sided_p']:.6f}. Gate={joint['gate_passed']}. Accepted pair jumps={joint['counts']['accepted_pair_jumps']}/128; removal old/new={100*joint['means']['old']['removed']:.3f}%/{100*joint['means']['joint']['removed']:.3f}%.", '',
        'An exact rational/algebraic counterexample proves that a certified single-deletion path can miss a feasible joint support under the same reference relation and support-dependent redistribution. This is an accessibility result, not a prediction theorem or a verified novelty claim. The real-data search reaches additional compact certified sets but has no accuracy advantage; same-old-count predictions are identical. The bounded pair pool is not a complete search.', '',
        '## Failure diagnostics', '', 'Native path: '+str(dict(counts)), '', 'Crossfit pilot: '+str(dict(failures)), '',
        f"Outer selected stages with zero original protection in a class: {len(zero_protected)} class-fold occurrences. Zero protection makes that class's retention constraint vacuous; it is not robust-classification evidence.", '',
        '## What is supported', '',
        '- Exact orthogonal reparameterization and fixed-center forest projection are independently verified; no measured runtime speedup or population bound is implied.',
        '- Every proposed support passes unchanged FRS dependency and majority checks.',
        '- Native and conditional exact-count controls share preprocessing, balanced classifier family, regularization search and perturbations.',
        '- All negative datasets, full-support selections, convergence caps and rare classes are retained.', '',
        '## Next step', '',
        'All tested predictive gates failed. Retain these implementations as explicit unsuccessful development experiments; do not select a favorable dataset or effect from them as confirmation. Certified joint accessibility is now established by an exact counterexample, but deeper reduction can harm prediction. The next complete nested study should allow train-internal choice across joint versus original paths and stopping depths, refit all selected models, and retain a full-feature option. Such nested selection may protect accuracy, but cannot be assumed to yield superiority. Faithful FD+/FCE+ reproduction and an untouched confirmation cohort remain outstanding.']
    (ROOT/'research/geometry-forest-iterations-review.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(result)


if __name__ == '__main__': main()
