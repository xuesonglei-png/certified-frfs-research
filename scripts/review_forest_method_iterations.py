"""Audited development history: retain failed gates and changed guarantees."""
import json
from review_box_iterations import ROOT


def main():
    identifiers = ('forest-penalty-selection', 'forest-compact-transfer', 'forest-aware-support', 'forest-teacher-guard-cost')
    saved = {}
    for key in identifiers:
        folder = ROOT/'results'/key
        assert (folder/'complete-audit.json').exists()
        saved[key] = json.loads((folder/'complete.json').read_text())
    rows = []
    for key, comparator in zip(identifiers, ('identical C1 full additive', 'identical support additive', 'old compact same classifier', 'latest guarded same support')):
        r = saved[key]
        rows.append({'experiment': key, 'comparator': comparator, 'gain_pp': r['clean_gain']['mean_pp'],
            'interval_pp': r['clean_gain']['bootstrap95_pp'], 'descriptive_p': r['clean_gain']['sign_flip_two_sided_p'],
            'gate_passed': r['gate_passed'], 'scope': r['scope'],
            'removed': r.get('removed', sum(row['removed'] for row in r['rows'])/15 if 'removed' in r['rows'][0] else 0.)})
    result = {'rows': rows, 'guard_counterfactual_counts': saved[identifiers[-1]]['counts'],
        'scope': 'four_audited_exposed_development_experiments_no_SOTA_or_confirmatory_superiority',
        'next_step': 'complete fresh nested FRS-certified selection with common downstream classifiers and matched recent native baselines; optional teacher guard must be explicitly separated from unchanged FRS guarantees'}
    (ROOT/'results/forest-method-iterations-review.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    lines = ['# Audited fuzzy forest method iteration review', '',
        'All four studies cover all15/all128 existing inner references and queries. All data are exposed development data; no original outer query labels enter these pilots. No LaTeX files were edited or compiled.', '',
        '| Experiment | Comparator | Clean gain (pp) | Descriptive sign-flip p | Actual removal | Resource gate |',
        '|---|---|---:|---:|---:|---|']
    lines += [f"| {r['experiment']} | {r['comparator']} | {r['gain_pp']:.4f} | {r['descriptive_p']:.6f} | {100*r['removed']:.2f}% | {r['gate_passed']} |" for r in rows]
    count = result['guard_counterfactual_counts']
    lines += ['', 'The first three gates failed. Interaction-aware deletion improved the old compact method by only.1203pp at exactly matching counts; it did not establish predictive superiority.', '',
        f"Disabling the auxiliary classifier teacher guard increased mean BA by{rows[-1]['gain_pp']:.4f}pp on identical supports, but the original full teacher guard failed in{count['full_teacher_guard_violations']}/{count['references']} references and90% per-class original protected-margin retention failed in{count['references_below_90percent_any_protected_class']} references. It must not inherit the former classifier guarantee.", '',
        'The underlying FRS geometry certificates remained verified on all128 identical supports. Thus the guard ablation identifies a cost of an additional classifier condition; it does not invalidate the FRS dependency/majority certificate. It is a diagnostic signal, not a new certified-classifier result. Its descriptive p-value does not establish statistically significant classification superiority.', '',
        'Next: explicitly distinguish FRS-certified attribute sets from teacher-margin-certified downstream classifiers. Build a fresh nested FRS-only selection/classification pipeline, keeping both variants available as a declared guarantee/performance tradeoff. Use identical classifier families and training-only searches for proposed/full/FGS/SFRIP, matched support counts and native stopping analyses. Resolve FD+/FCE+ fidelity before complete modern-baseline claims, and freeze before eventual independent confirmation.', '',
        'No outcome-based dataset exclusion, test-selected penalty, or posthoc removal of failed attempts was used. All previous variants and negative outcomes are retained. None of these four development pilots satisfies the active publication-level superiority goal.']
    (ROOT/'research/forest-method-iterations-review.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(result, flush=True)


if __name__ == '__main__': main()
