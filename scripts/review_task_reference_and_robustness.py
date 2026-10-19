"""Audited successes and failures, exact scientific plot data, no TeX edits."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_nested_joint_path import ROOT,digest


def verified(folder):
    summary = json.loads((folder/'complete.json').read_text()); audit = json.loads((folder/'complete-audit.json').read_text())
    assert audit['passed']
    for row in summary['rows']:
        name = row['dataset']; checkpoint = json.loads((folder/(name+'-audit.json')).read_text())
        if 'artifact_sha256' in checkpoint.get('fingerprints',{}):
            assert checkpoint['fingerprints']['artifact_sha256'] == digest(folder/(name+'.json'))
    return summary,audit


def main():
    task,taskaudit = verified(ROOT/'results/task-reference-geometry-pilot')
    curves,curveaudit = verified(ROOT/'results/forest-robustness-curve')
    nested,nestedaudit = verified(ROOT/'results/nested-robust-policy')
    names = [r['dataset'] for r in curves['rows']]
    nonzero = {'task_reference':[],'nested_selection':[]}; unchanged_normalization = 0
    for name in names:
        taskfile = json.loads((ROOT/'results/task-reference-geometry-pilot'/(name+'.json')).read_text())
        for r in taskfile['records']:
            unchanged_normalization += r['fits']['uniform_native']['features'] == r['fits']['old']['features']
            delta = r['metrics']['task_native']['clean']-r['metrics']['old']['clean']
            if abs(delta) > 1e-12:
                nonzero['task_reference'].append({'dataset':name,'outer_index':r['outer_index'],'inner_index':r['inner_index'],
                    'clean_gain_pp':100*delta,'old_features':r['fits']['old']['features'],
                    'task_features':r['fits']['task_native']['features'],'matched_count_exact':r['count_exact']})
        saved = json.loads((ROOT/'results/nested-robust-policy'/(name+'.json')).read_text())
        for i,s in enumerate(saved['splits']):
            old,new = s['results']['old_proposed'],s['results']['proposed']
            dc = new['metrics']['clean']-old['metrics']['clean']; dr = new['metrics']['robust']-old['metrics']['robust']
            if abs(dc)+abs(dr) > 1e-12:
                nonzero['nested_selection'].append({'dataset':name,'outer_index':i,'clean_gain_pp':100*dc,
                    'robust_gain_pp':100*dr,'old_features':old['features'],'new_features':new['features'],
                    'clean_loss_despite_inner_band':dc < -.005-1e-12})
    assert unchanged_normalization == 128
    report = {'task_reference':task,'task_reference_audit':taskaudit,'robustness_curve':curves,
        'robustness_curve_audit':curveaudit,'nested_selection':nested,'nested_selection_audit':nestedaudit,
        'all_nonzero_cases':nonzero,'goal_achieved':False,
        'scope':'all_successes_failures_exposed_development_no_clean_accuracy_SOTA_or_novelty_assertion'}
    (ROOT/'results/task-reference-robustness-review.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    primary = curves['same_classifier_certificate_gain']['proposed']; selection = nested['selection_effects']['proposed']
    lines = ['# Task reference and exact robustness: audited iteration review','',
        '2026-10-10. Three experiments and all independent audits complete. No TeX edits or compilation.',
        'Goal remains active and unachieved. All datasets/partitions are repeatedly exposed development data.','',
        '## A practical benefit of the exact certificate','',
        'Same fitted classifiers, same query labels, same numeric boxes. Fixed radii0,.01,.03,.05,.10; area is the normalized trapezoid integral over0..10%.',
        f"Proposed exact/independent/Lipschitz certified areas: {100*curves['means']['proposed']['area']['exact']:.4f}% / {100*curves['means']['proposed']['area']['independent']:.4f}% / {100*curves['means']['proposed']['area']['lipschitz']:.4f}%.",'',
        '|Certificate comparison on identical proposed fits|Gain pp|Descriptive95% pp|Sign-flip p|W/L/T|',
        '|---|---:|---|---:|---|']
    for k,v in primary.items():
        lines.append(f"|exact versus {k}|{v['mean_pp']:.4f}|[{v['bootstrap95_pp'][0]:.4f}, {v['bootstrap95_pp'][1]:.4f}]|{v['sign_flip_two_sided_p']:.8f}|{v['wins']}/{v['losses']}/{v['ties']}|")
    lines += ['',
        'Exact dominance is a deterministic mathematical property; its descriptive p-value is not independent evidence of algorithmic novelty. The measured magnitude shows utility in avoiding false negatives of conservative certification. It does not change a single clean prediction.',
        f"Audit: {curveaudit['counts']['independent_leaf_elimination_replays']} all-query independent leaf-elimination replays, {curveaudit['counts']['independent_lp_solves']} independent integral marginal LP solves, {curveaudit['counts']['witness_score_replays']} raw witness score evaluations. Tightening LP solver feasibility tolerances resolved a5.08e-8 numerical oracle discrepancy without changing experiment results.",'',
        '## Native and count-adapted method robustness','',
        '|Exact curve area: proposed minus comparator|Gain pp|Descriptive p|Holm p|',
        '|---|---:|---:|---:|']
    for m,v in curves['exact_method_gain'].items():
        lines.append(f"|{m}|{v['mean_pp']:.4f}|{v['sign_flip_two_sided_p']:.6f}|{curves['exact_method_gain_holm'][m]:.6f}|")
    lines += ['',
        'No native/matched method contrast passes Holm correction. Native SFRIP certifies a larger curve area while removing more attributes and has lower clean accuracy. Matched controls are conditional adaptations with proposed-selected budgets, not original native stopping. Therefore improved certification tightness cannot be substituted for recent-method prediction or robustness superiority.','',
        '## Failed task-informed relation pilot','',
        f"Task-native clean gain versus old: {task['contrasts']['clean']['task_native']['mean_pp']:.4f}pp, p={task['contrasts']['clean']['task_native']['sign_flip_two_sided_p']:.6f}. Old/new removal {100*task['means']['old']['removed']:.4f}%/{100*task['means']['task_native']['removed']:.4f}%. Gate={task['gate_passed']}.",
        f"Matched clean gain {task['contrasts']['clean']['task_matched']['mean_pp']:.4f}pp, {task['counts']['count_mismatches']} count mismatches. Uniform unit-mass ablation produces the SAME128 supports/predictions as old. Learned worst-case sensitivities change some supports, but do not improve dataset-average clean BA.",
        'Reference relations and protected masks change explicitly. All admitted supports retain general-weight FRS dependency/majority certificates relative to their NEW reference, not the old relation. Fixed-reference-teacher sensitivities do not certify an independently refitted classifier.',
        'An unreachable failed-gate summary-key typo was corrected after execution; the exact executed v1 script is preserved in research/experiment-source. Only the p-value key and fingerprint provenance changed, verified by an exact byte-level diff; every fit, support, score, and scientific parameter was unchanged, and the audit was rerun.','',
        '## Failed use of exact robustness in common inner selection','',
        f"All methods received the same .5pp inner clean band and exact radius.03 criterion. {nested['counts']['inner_score_models']} archived inner models re-evaluated; {nested['counts']['selected_proposed_changed']}/44 proposed selections changed.",
        f"New versus old proposed: clean {selection['clean']['mean_pp']:.4f}pp, robust {selection['robust']['mean_pp']:.4f}pp (p={selection['robust']['sign_flip_two_sided_p']:.6f}), curve area {selection['area']['mean_pp']:.4f}pp. Gate={nested['gate_passed']}.",
        f"Independent audit verified all128 inner folds, all44 proposed certificates, {nestedaudit['counts']['outer_lp_solves']} outer LP solves and {nestedaudit['counts']['independent_classifier_refits']} fresh outer classifier refits; all remaining outer models have exact source-audit provenance.",
        'The inner clean band is not held-out noninferiority; adverse outer cases are explicitly archived. Repeating candidate searches on these data cannot turn them into independent confirmation.','',
        '## Research decision','',
        'Retain exact certification as an experimentally supported contribution with a precise fixed-model scope. Do not promote either new relation or robustness selection as a successful prediction method. Do not tune the radius, tolerance, fraction, or significance test from these query results.',
        'Next hypothesis to freeze before any new result: check whether the fixed three-knot triangular representation and coarse interaction discretization constrain useful nonlinear classification, with a generalized piecewise-linear forest and equal classifier capacity for all baseline supports. Exact forest inference extends to endpoint/interior-knot states; any extra capacity must be separated from feature-selection gains.',
        'Before a high-level journal/SOTA claim: faithful FD+/FCE+ conventions, comparable native and actual-count controls, untouched confirmation data, and novelty comparison against established tree graphical inference/certification. Mere tighter margins do not establish novelty of generic min-sum recursion or universal feature-selection superiority.']
    (ROOT/'research/task-reference-robustness-review.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    folder = ROOT/'results/figures/method-development'; folder.mkdir(parents=True,exist_ok=True)
    data = {'radii':curves['radii'],'means':curves['means'],
        'per_dataset_certificate_gain':{r['dataset']:{k:100*(r['methods']['proposed']['area']['exact']-r['methods']['proposed']['area'][k]) for k in ('independent','lipschitz')} for r in curves['rows']},
        'scope':'same_fit_certification_benefit_no_general_prediction_superiority_exposed_development'}
    (folder/'exact-certification-benefit-data.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    fig,axes = plt.subplots(1,3,figsize=(16,6),gridspec_kw={'width_ratios':[1,1.05,1.2]})
    colors = {'exact':'#2459a6','independent':'#e09028','lipschitz':'#737a83'}
    for k,label in [('exact','Exact forest'),('independent','Independent terms'),('lipschitz','Coordinate Lipschitz')]:
        axes[0].plot(100*np.array(curves['radii']),100*np.array(curves['means']['proposed']['curves'][k]),'o-',color=colors[k],label=label)
    axes[0].set(title='Same classifier, different certificates',xlabel='Numeric box radius (% of training range)',ylabel='Class-balanced certified coverage (%)')
    axes[0].legend(frameon=False,fontsize=9); axes[0].set_ylim(0,85)
    for k,marker,color,label in [('independent','o',colors['independent'],'Exact minus independent'),('lipschitz','s',colors['lipschitz'],'Exact minus Lipschitz')]:
        values = [data['per_dataset_certificate_gain'][name][k] for name in names]
        axes[1].plot(values,np.arange(len(names)),marker,ls='none',color=color,label=label)
    axes[1].set_yticks(np.arange(len(names)),names,fontsize=9); axes[1].invert_yaxis()
    axes[1].set(title='Certified curve-area gain',xlabel='Difference (percentage points)'); axes[1].axvline(0,color='#aaa',lw=1)
    axes[1].legend(frameon=False,fontsize=8,loc='lower right')
    palette = {'proposed':'#2459a6','full':'#737a83','fgs2023':'#c46b32','sfrip2025':'#2d8e77'}
    for m,label in [('proposed','Proposed'),('full','Full attributes'),('fgs2023','FGS native'),('sfrip2025','SFRIP native')]:
        removed = 100*curves['means'][m]['removed']
        axes[2].plot(100*np.array(curves['radii']),100*np.array(curves['means'][m]['curves']['exact']),'o-',color=palette[m],
            label=f'{label} ({removed:.1f}% removed)')
    axes[2].set(title='Method comparison with exact certificates',xlabel='Numeric box radius (% of training range)',ylabel='Class-balanced certified coverage (%)')
    axes[2].set_ylim(0,85); axes[2].legend(frameon=False,fontsize=8)
    for ax in axes:
        ax.grid(axis='x' if ax is axes[1] else 'both',alpha=.2); ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Exact certificates add coverage; method superiority remains unestablished',fontsize=14,y=.985)
    fig.text(.5,.024,'Means over 15 exposed development datasets; fixed numeric boxes. Certificate gains do not change clean predictions.\nNative budgets differ; no native or count-adapted method-area comparison survives Holm correction.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.09,1,.95),w_pad=2.4)
    fig.savefig(folder/'exact-certification-benefit.png',dpi=180); fig.savefig(folder/'exact-certification-benefit.svg'); plt.close(fig)
    print({'task_gate':task['gate_passed'],'certificate_gate':curves['diagnostic_gate_passed'],
        'nested_robust_gate':nested['gate_passed'],'goal_achieved':False,'nonzero_cases':{k:len(v) for k,v in nonzero.items()}},flush=True)


if __name__ == '__main__': main()
