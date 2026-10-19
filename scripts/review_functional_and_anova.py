"""All audited capacity, normalization and surrogate-DP outcomes together."""
import json
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
        assert checkpoint['fingerprints']['artifact_sha256'] == digest(folder/(name+'.json'))
    return summary,audit


def main():
    capacity,capacityaudit = verified(ROOT/'results/spline-capacity-pilot')
    functional,functionalaudit = verified(ROOT/'results/functional-normalization-pilot')
    dp,dpaudit = verified(ROOT/'results/forest-anova-dp-pilot')
    caps = json.loads((ROOT/'results/functional-normalization-optimizer-caps.json').read_text())
    assert len(caps['caps']) == functional['counts']['child_optimizer_caps'] == 3
    assert all(not r['selected_by'] for r in caps['caps'])
    names = [r['dataset'] for r in functional['rows']]
    allcases = {'functional':[],'DP':[]}; countchecks = {'dp_more_attributes_than_old':0,'dp_less_attributes_than_old':0,
        'dp_same_count_as_old':0,'same_count_reconstruction_improves':0,'same_count_reconstruction_improves_query_declines':0}
    for name in names:
        file = json.loads((ROOT/'results/functional-normalization-pilot'/(name+'.json')).read_text())
        for r in file['records']:
            gains = {m:100*(r['metrics']['certified']['clean']-r['metrics'][m]['clean']) for m in ('old3','full','fgs_matched','sfrip_matched')}
            if any(abs(v) > 1e-12 for v in gains.values()):
                allcases['functional'].append({'dataset':name,'outer_index':r['outer_index'],'inner_index':r['inner_index'],
                    'clean_gains_pp':gains,'selected_configs':r['selected_configs'],'supports':r['supports']})
        file = json.loads((ROOT/'results/forest-anova-dp-pilot'/(name+'.json')).read_text())
        for r in file['records']:
            old,new = len(r['fits']['old']['features']),r['target_count']
            key = 'dp_more_attributes_than_old' if new > old else 'dp_less_attributes_than_old' if new < old else 'dp_same_count_as_old'
            countchecks[key] += 1
            improves = r['uniform_omission_losses']['selected_dp'] < r['uniform_omission_losses']['old']-1e-12
            if new == old and improves:
                countchecks['same_count_reconstruction_improves'] += 1
                countchecks['same_count_reconstruction_improves_query_declines'] += r['metrics']['dp_refit']['clean'] < r['metrics']['old']['clean']-1e-12
            gains = {m:100*(r['metrics']['dp_refit']['clean']-r['metrics'][m]['clean']) for m in ('old','full','fgs_matched','sfrip_matched')}
            if any(abs(v) > 1e-12 for v in gains.values()):
                allcases['DP'].append({'dataset':name,'outer_index':r['outer_index'],'inner_index':r['inner_index'],
                    'clean_gains_pp':gains,'old_count':old,'new_count':new,'omission_loss':r['uniform_omission_losses'],
                    'old_features':r['fits']['old']['features'],'new_features':r['fits']['dp_refit']['features'],
                    'attempt_counts':[c['dp']['count'] for c in r['candidates']]})
    report = {'capacity':capacity,'capacity_audit':capacityaudit,'functional':functional,'functional_audit':functionalaudit,
        'DP':dp,'DP_audit':dpaudit,'optimizer_caps':caps,'count_diagnostics':countchecks,'all_nonzero_cases':allcases,
        'goal_achieved':False,'scope':'exposed_development_three_unsuccessful_prediction_directions_distinct_theory_value'}
    (ROOT/'results/functional-and-anova-review.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines = ['# Functional capacity, normalization and exact surrogate DP: audited review','',
        '2026-10-10. All three new experiments and independent audits complete. No TeX edits or compilation.',
        'Goal remains ACTIVE/UNACHIEVED. These repeatedly exposed development data do not establish confirmation, general predictive advantage, or recent SOTA.','',
        '## Conclusions','',
        'The fixed five-knot nodal penalty comparison failed. Functional normalization plus complete training-internal pipeline tuning also failed. Exact uniform-domain reconstruction DP marginally improves accuracy over the old method while retaining more attributes; same-count recent controls do not establish superior feature quality. These distinctions prevent a misleading positive conclusion.','',
        '|Method/pilot|Clean BA %|Actual removal %|Gain over its old method pp|Equal-count FGS gain pp|Equal-count SFRIP gain pp|',
        '|---|---:|---:|---:|---:|---:|',
        f"|Five-knot nodal search|{100*capacity['means']['proposed5']['clean']:.4f}|{100*capacity['means']['proposed5']['removed']:.4f}|{capacity['ablations']['clean']['new_vs_old']['mean_pp']:.4f}|{capacity['method_contrasts']['clean']['fgs_matched5']['mean_pp']:.4f}|{capacity['method_contrasts']['clean']['sfrip_matched5']['mean_pp']:.4f}|",
        f"|Functional norm + full internal CV|{100*functional['means']['certified']['clean']:.4f}|{100*functional['means']['certified']['removed']:.4f}|{functional['contrasts']['clean']['old3']['mean_pp']:.4f}|{functional['contrasts']['clean']['fgs_matched']['mean_pp']:.4f}|{functional['contrasts']['clean']['sfrip_matched']['mean_pp']:.4f}|",
        f"|Exact ANOVA DP + FRS screen|{100*dp['means']['dp_refit']['clean']:.4f}|{100*dp['means']['dp_refit']['removed']:.4f}|{dp['contrasts']['clean']['old']['mean_pp']:.4f}|{dp['contrasts']['clean']['fgs_matched']['mean_pp']:.4f}|{dp['contrasts']['clean']['sfrip_matched']['mean_pp']:.4f}|",'',
        '## Functional-normalization full pipeline audit','',
        f"{functional['counts']['child_folds']} internal training folds; {functional['counts']['child_models']} candidate models independently refit. Each child trains its own preprocessing/selectors/trees; validation labels enter only classifier scores. Parent actual counts match on all128 reference/query cases.",
        f"Audit precision maxima: {functionalaudit['precision_maxima']}. Native recent configurations are fixed pilot settings; selectors do not receive complete independent grid tuning. No faithfulFD+/FCE+ or definitive modern-SOTA comparison.",
        'Three child models reached4000 iterations, all on Yeast5knotsC10/pair1. None was selected by any pipeline. Independent gradient infinities4.84e-7,2.14e-7,1.31e-6 are recorded separately; optimizer flags remainFalse. The frozen gate requires zero caps, so they are not ignored or relabeled converged. The predictive gate fails on comparative effects independently of these caps.','',
        '## Exact forest surrogate DP audit and limitations','',
        f"{dpaudit['counts']['independent_exhaustive_optima']} exhaustive cardinality optima and {dpaudit['counts'].get('independent_MILP_optima',0)} independent global MILP optima verify the surrogate. {dp['counts']['total_dp_candidates']} attempted candidate counts were screened against original full-reference FRS certificates.",
        f"{dp['counts']['restored_count_references']} references increase requested count to find a passing DP candidate; {dp['counts']['full_fallbacks']} deploy all attributes. Count diagnostics: {countchecks}.",
        'The DP minimizes a DECLARED uniform-domain fixed-model reconstruction loss. It does not optimize test error, does not solve the global FRS-feasible optimization, and does not imply that an independently refitted classifier preserves a teacher margin. Native controls have different counts; matched controls share the actual accepted count.',
        'Canonical raw-ANOVA conversion preserves arbitrary original score functions and exact raw boxes. Independent omitted-coordinate quadrature reproduces the projected fixed-teacher scores. Numerical classifier refits, independent forest leaf elimination, integral marginal LP and attained witnesses verify the deployed results.','',
        '## Theory/code retained','',
        '1. Arbitrary-knot fuzzy partitions admit exact endpoint/interior-knot forest box separation;3-knot compatibility is verified.',
        '2. Exact Gram/refinement identities expose a nodal-penalty confound. A function-space norm makes refinement penalties and projected scores invariant. Functional ANOVA and spline normalization are established ideas; no standalone novelty claim.',
        '3. Uniform-domain conditional projection has exact omitted unary/incident-edge energy. Forest knapsack DP globally minimizes this surrogate at fixed count. A disagreement bound additionally requires explicit distribution/margin assumptions, unverified for UCI; it is not an empirical prediction guarantee.',
        '4. Earlier exact-certification coverage improvement remains supported but does not change classifier predictions or establish SOTA feature selection.','',
        '## Next research decision','',
        'Stop treating classifier capacity, relation normalization or better surrogate optimization as automatic accuracy improvements. On these data their controlled comparisons are negative. Any further method change needs an explicit new mechanism and a frozen common-capacity matched-count design, not more query-driven parameter searches.',
        'Investigate whether training-population geometry can justify replacing the uniform-domain reconstruction criterion, with exact empirical/global optimization and independent certification; distinguish training-optimal reconstruction from classification. Review feasibility rejections and class-specific certificate contributions before changing certificate targets. Preserve the original target and do not discard guarantees merely to obtain a positive plot.',
        'Complete faithful recent baseline conventions (especiallyFD+/FCE+) and prepare untouched confirmation before a high-level journal claim. No manuscript wording should imply that these failed directions meet the requested significance standard.']
    (ROOT/'research/functional-and-anova-review.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    folder = ROOT/'results/figures/method-development'; folder.mkdir(parents=True,exist_ok=True)
    data = {'datasets':names,'pilots':{},'scope':'all15_exposed_development_matched_actual_counts_no_SOTA_claim'}
    definitions = [('Five-knot nodal',capacity,'proposed5',('fgs_matched5','sfrip_matched5')),
        ('Functional + internal CV',functional,'certified',('fgs_matched','sfrip_matched')),
        ('Exact ANOVA DP',dp,'dp_refit',('fgs_matched','sfrip_matched'))]
    for label,summary,method,controls in definitions:
        lookup = {r['dataset']:r for r in summary['rows']}
        data['pilots'][label] = {c:[100*(lookup[n]['methods'][method]['clean']-lookup[n]['methods'][c]['clean']) for n in names] for c in controls}
    (folder/'capacity-normalization-dp-data.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    fig,axes = plt.subplots(1,3,figsize=(15,6.3),sharey=True)
    colors = ['#b66b31','#2c698e']
    for ax,(label,summary,method,controls) in zip(axes,definitions):
        for c,color,offset,legend in zip(controls,colors,(-.12,.12),('versus same-count FGS','versus same-count SFRIP')):
            ax.plot(data['pilots'][label][c],np.arange(len(names))+offset,'o',ms=5,color=color,label=legend)
        removed = 100*summary['means'][method]['removed']
        ax.axvline(0,color='#777',lw=1); ax.set_title(f'{label}\n{removed:.2f}% attributes removed',fontsize=11)
        ax.set_xlabel('Clean balanced-accuracy difference (pp)'); ax.grid(axis='x',alpha=.2)
        ax.spines[['top','right']].set_visible(False); ax.set_xlim(-8,8)
    axes[0].set_yticks(np.arange(len(names)),names,fontsize=9); axes[0].invert_yaxis(); axes[-1].legend(frameon=False,fontsize=9,loc='lower right')
    fig.suptitle('Three controlled method iterations: no verified same-count predictive superiority',fontsize=13,y=.98)
    fig.text(.5,.027,'Every dataset retained; differences averaged within dataset. Reference/query data are exposed development.\nEach column has its own actual support budget, matched exactly by both controls; native stopping is evaluated separately.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.10,1,.94),w_pad=2)
    fig.savefig(folder/'capacity-normalization-dp.png',dpi=180); fig.savefig(folder/'capacity-normalization-dp.svg'); plt.close(fig)
    print({'all_three_prediction_gates_failed':True,'count_diagnostics':countchecks,'goal_achieved':False},flush=True)


if __name__ == '__main__': main()
