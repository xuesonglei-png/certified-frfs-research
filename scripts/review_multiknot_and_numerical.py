"""Verify immutable evidence and plot all sample-depth diagnostic outcomes."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_nested_joint_path import ROOT, digest


if __name__ == '__main__':
    mechanism_folder = ROOT/'results/multiknot-forest-mechanism-screen'
    depth_folder = ROOT/'results/forest-sample-depth-diagnosis'
    replay_folder = ROOT/'results/forest-sample-depth-reproduction'
    mechanism = json.loads((mechanism_folder/'complete.json').read_text())
    depth = json.loads((depth_folder/'complete.json').read_text())
    evidence = [ROOT/'results/robust-logloss-numerical-diagnoses-audit.json',
                ROOT/'results/development-sample-depth-audit.json',
                mechanism_folder/'complete-audit.json', depth_folder/'complete-audit.json',
                replay_folder/'complete-audit.json', replay_folder/'reproduction-equivalence-audit.json']
    hashes = {}
    for path in evidence:
        saved = json.loads(path.read_text())
        assert saved['passed']
        hashes[path.relative_to(ROOT).as_posix()] = digest(path)
        if 'auditor_sha256' in saved:
            auditor = {evidence[0]: ROOT/'scripts/audit_robust_logloss_numerical_diagnoses.py',
                       evidence[1]: ROOT/'scripts/audit_development_sample_depth.py',
                       evidence[2]: ROOT/'scripts/audit_multiknot_forest_screen.py',
                       evidence[3]: ROOT/'scripts/audit_forest_sample_depth.py',
                       evidence[4]: ROOT/'scripts/audit_forest_sample_depth.py',
                       evidence[5]: ROOT/'scripts/audit_forest_sample_depth_reproduction.py'}[path]
            assert digest(auditor) == saved['auditor_sha256']
        for name, sha in saved.get('artifact_hashes', {}).items():
            assert digest(path.parent/(name+'.json')) == sha
        if 'summary_sha256' in saved:
            assert digest(path.parent/'complete.json') == saved['summary_sha256']
        for rel, sha in saved.get('fingerprints', {}).items():
            assert digest(ROOT/rel) == sha
    numerical = json.loads(evidence[0].read_text())
    for row in numerical['rows']:
        path = ROOT/'results'/(row['diagnosis']+'.json')
        assert digest(path) == row['artifact_sha256']
        data = json.loads(path.read_text())
        assert not data['validation_scores_used']
        for rel, sha in data['fingerprints'].items():
            assert digest(ROOT/rel) == sha
    assert not mechanism['numeric_resource_gate'] and not mechanism['goal_achieved']
    assert not depth['goal_achieved'] and not depth['support_selectors_evaluated']
    fractions = ['0.3', '0.6', '1.0']
    x = np.array([.3, .6, 1.])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout='constrained')
    for key, color in [('3', '#2864a5'), ('5', '#ce6519')]:
        clean = [100*depth['means'][f][key]['clean_BA'] for f in fractions]
        robust = [100*depth['means'][f][key]['certified_BA'] for f in fractions]
        axes[0].plot(x, clean, '-o', color=color, label=key+' knots: clean')
        axes[0].plot(x, robust, '--s', color=color, label=key+' knots: certified')
    axes[0].set(xlabel='Fraction of outer training set', ylabel='Mean balanced accuracy (%)',
                title='All 15 datasets; all 3 outer folds')
    axes[0].legend(fontsize=8)
    effects = [depth['capacity_gain_by_fraction'][f] for f in fractions]
    central = np.array([v['mean_pp'] for v in effects])
    lower, upper = np.array([v['bootstrap95_pp'] for v in effects]).T
    axes[1].errorbar(x, central, yerr=np.stack([central-lower, upper-central]), fmt='o-', color='#644897', capsize=5)
    axes[1].axhline(0., color='#777777', lw=1.)
    axes[1].set(xlabel='Fraction of outer training set', ylabel='5 minus 3 knots (percentage points)',
                title='Clean accuracy contrast; descriptive 95% intervals')
    for ax in axes:
        ax.set_xticks(x)
        ax.grid(alpha=.2)
    output = ROOT/'output'
    output.mkdir(exist_ok=True)
    png = output/'forest-sample-depth-development.png'
    fig.savefig(png, dpi=180)
    plt.close(fig)
    suite = ROOT/'results/multiknot-and-numerical-suite.log'
    assert '676 passed, 3 warnings' in suite.read_text(encoding='utf-8', errors='replace')
    lines = ['# Numerical repair, knot capacity and sample-depth review', '',
        '2026-10-10. Goal ACTIVE / UNACHIEVED. No LaTeX editing or compilation.', '',
        'No significant predictive superiority over recent feature-selection baselines has been established.', '',
        '## Numerical repair', '',
        'Removing redundant own-class nominal inequalities recovers5/7 previously failed training folds;2remain failed. '
        'Positive objective scaling separately recovers3/7. Do not combine the variants or count union recoveries as another tested model. '
        'Both diagnoses pass independent legacy-DP, coefficient/objective, cut and LP audits.114 LP probes; '
        'no validation scores were computed and the original robust-risk classifier gate stays failed.', '',
        '## Five-knot mechanism', '',
        'The new piecewise-linear fuzzy forest supports uniform/nonuniform knots, an orthogonal classifier and exact shared-coordinate box margins. '
        'Tests cover legacy3 equivalence, exhaustive raw-grid extrema, refined-family containment and invalid cycles. '
        'Refinement expands the score family; its L2 penalty changes, and accuracy need not improve. '
        'This is an elementary representation extension, not a verified novel theorem.', '',
        f"All15/all44 internal development folds: clean BA gain {mechanism['clean_BA_gain_pp']:.6f}pp,8wins/6losses/1tie. "
        'The fixed0.5pp gate FAILS; uncertainty crosses zero. No threshold relaxation or all128 expansion. '
        'Fresh88-fit audit and78LP probes pass.', '',
        '## Verified sample depth', '',
        'The archived diagnostic references descend from iteration41 original outer training, its inner references, '
        'another nested reference, then the mechanism fold. Sonar fits use40/41 of208 raw observations. '
        'This is nested development provenance, not a hidden cap or proven bug. '
        'The original outer split fixes3folds even for rare classes; retain that documented convention.', '',
        'A separate frozen learning-curve diagnosis evaluates3fixed training budgets on ALL15/all45 original outer folds. '
        'Both classifier arms share each training-only preprocessing and learned forest. Full attributes throughout. '
        'All270 fits complete without iteration caps and are independently refit/audited, with45312witness replays and78LP probes.', '',
        '| Training fraction | Three-knot clean BA | Five-knot clean BA | Difference (pp) | Descriptive 95% interval (pp) |',
        '|---|---:|---:|---:|---:|']
    for f in fractions:
        effect = depth['capacity_gain_by_fraction'][f]
        lo, hi = effect['bootstrap95_pp']
        lines.append(f"| {f} | {100*depth['means'][f]['3']['clean_BA']:.4f}% | {100*depth['means'][f]['5']['clean_BA']:.4f}% | {effect['mean_pp']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines += ['', 'PRIMARY diagnostic interaction (full-budget capacity contrast minus small-budget contrast) is+1.784160pp. '
        'This supports investigating sample-dependent capacity, but all data are exposed and this is not confirmatory significance. '
        'The full-budget clean difference remains inconclusive:7wins/7losses/1tie, interval crosses zero. '
        'Full-budget certified BA drops from77.2520% to75.9703%; capacity has a robustness cost.', '',
        '## Reproduction repair', '',
        'The frozen sample-depth runner accumulated parent fingerprints during execution, making completed-cache reuse reject a partial fingerprint dictionary. '
        'A preserved-source v2 computes all parent hashes before checkpoint validation. All270models and all predictions reproduce the original exactly, '
        'then all15 completed-cache replays pass. Original sources/artifacts remain immutable; this is a code repair, not another effect estimate.', '',
        '## Remaining research requirement', '',
        'Recent selectors were not run in these model diagnostics. FD+/FCE+ relation/operator conventions remain unresolved. '
        'Next predictive comparisons must use proper raw-data outer training budgets, refit all supports inside inner training, '
        'give each selector the same downstream model search, report actual feature counts and native stopping separately, '
        'and retain a fresh confirmation set. Selecting knots or regularization by the outer learning-curve scores would be leakage. '
        'Do not present these findings as achieved reduct superiority, test-risk certification or publication readiness.', '',
        'Full production-module regression suite:676passed,3warnings. All experiment/audit/test processes terminal. '
        'Figure: output/forest-sample-depth-development.png.']
    (ROOT/'research/multiknot-and-sample-depth-review.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    result = {'passed': True, 'fingerprints': hashes, 'reviewer_sha256': digest(Path(__file__)),
              'goal_achieved': False, 'main_prediction_superiority_established': False,
              'mechanism_gate_passed': False, 'suite_passed': 676,
              'full_budget_clean_gain_pp': depth['capacity_gain_by_fraction']['1.0']['mean_pp'],
              'primary_development_interaction_pp': depth['primary_sample_depth_interaction']['mean_pp'],
              'figure_sha256': digest(png), 'review_sha256': digest(ROOT/'research/multiknot-and-sample-depth-review.md')}
    (ROOT/'results/multiknot-and-sample-depth-review.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(result, flush=True)
