"""All latest directions, including the failed functional audit, without filtering."""
import json
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_nested_joint_path import ROOT,digest

def read(folder): return json.loads((ROOT/'results'/folder/'complete.json').read_text())

def verified(folder):
    path = ROOT/'results'/folder; summary = read(folder); audit = json.loads((path/'complete-audit.json').read_text()); assert audit['passed']
    for row in summary['rows']:
        name = row['dataset']; checkpoint = json.loads((path/(name+'-audit.json')).read_text())
        fingerprint = checkpoint['fingerprints'].get('artifact_sha256',checkpoint['fingerprints'].get(name+'.json'))
        assert fingerprint == digest(path/(name+'.json'))
    return summary,audit

def main():
    dp,dpaudit = verified('forest-anova-dp-pilot'); empirical,empaudit = verified('empirical-product-fixed-count')
    sobolev,sobaudit = verified('sobolev-classifier-pilot'); functional = read('functional-normalization-pilot')
    failure = json.loads((ROOT/'results/functional-audit-failure-diagnosis.json').read_text()); assert not failure['audit_passed']
    numeric = json.loads((ROOT/'results/sobolev-stationarity-diagnosis.json').read_text())
    assert numeric['code_sha256'] == digest(ROOT/'src/certified_frfs/logistic_stationarity_refinement.py')
    rejected = Counter(); cases = []; distributions = []
    for row in empirical['rows']:
        saved = json.loads((ROOT/'results/empirical-product-fixed-count'/(row['dataset']+'.json')).read_text())
        for r in saved['records']:
            path = r['paths']['empirical']
            for attempt in path['attempts']:
                if attempt['accepted']: continue
                cert = attempt['candidate']['certificate']; majority = not cert['majority_feasible']; dependency = cert['dependency_certificate']['upper_bound'] > .1
                assert majority or dependency
                rejected['both' if majority and dependency else 'majority_only' if majority else 'dependency_only'] += 1
            gains = {m:100*(r['metrics']['empirical_refit']['clean']-r['metrics'][m]['clean']) for m in ('old','full','fgs_matched','sfrip_matched')}
            cases.append({'dataset':row['dataset'],'outer':r['outer_index'],'inner':r['inner_index'],
                'support_changed':path['features'] != path['initial_features'],'surrogate_gain':path['initial_loss']-path['final_loss'],
                'clean_gains_pp':gains,'features':path['features'],'initial_features':path['initial_features']})
        distributions.append({'dataset':row['dataset'],**{m:100*(row['methods']['empirical_refit']['clean']-row['methods'][m]['clean']) for m in ('old','full','fgs_matched','sfrip_matched')}})
    all_sob_cases = []
    for row in sobolev['rows']:
        saved = json.loads((ROOT/'results/sobolev-classifier-pilot'/(row['dataset']+'.json')).read_text())
        for r in saved['records']:
            all_sob_cases.append({'dataset':row['dataset'],'outer':r['outer_index'],'choices':r['choices'],'supports':r['supports'],
                'clean_gains_pp':{m:100*(r['metrics']['refined']['clean']-r['metrics'][m]['clean']) for m in ('old_selector','full','fgs_matched','sfrip_matched','refined_alpha0')}})
    report = {'goal_achieved':False,'DP':dp,'DP_audit':dpaudit,'empirical':empirical,'empirical_audit':empaudit,
        'sobolev':sobolev,'sobolev_audit':sobaudit,'functional_unaccepted':functional,'functional_audit_failure':failure,
        'stationarity_diagnosis':numeric,'empirical_rejection_causes':dict(rejected),
        'all_empirical_cases':cases,'all_sobolev_cases':all_sob_cases,'empirical_dataset_gains':distributions,
        'scope':'complete_negative_development_evidence_not_prediction_superiority_or_confirmation'}
    (ROOT/'results/latest-method-development-review.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    entries = [
        ('Uniform DP, increasing count',dp['means']['dp_refit']['removed'],dp['contrasts']['clean']),
        ('Empirical product, fixed count',empirical['means']['empirical_refit']['removed'],empirical['contrasts']['clean']),
        ('Sobolev + internal CV',sobolev['means']['refined']['removed'],sobolev['contrasts']['clean'])]
    figure,axes = plt.subplots(1,2,figsize=(11,4.8),gridspec_kw={'width_ratios':[1.8,1]})
    y = np.arange(len(entries)); colors = ['#566c86','#007b83','#ba5b36']; comparators = [('full','Full'),('fgs_matched','FGS, equal count'),('sfrip_matched','SFRIP, equal count')]
    for ci,(key,label) in enumerate(comparators):
        vals = [e[2][key]['mean_pp'] for e in entries]; intervals = [e[2][key]['bootstrap95_pp'] for e in entries]
        error = np.array([[v-b[0] for v,b in zip(vals,intervals)],[b[1]-v for v,b in zip(vals,intervals)]])
        axes[0].errorbar(vals,y+(ci-1)*.2,xerr=error,fmt='o',capsize=3,label=label,color=colors[ci],ms=5)
    axes[0].axvline(0,color='#444',lw=1); axes[0].set_yticks(y,[e[0] for e in entries]); axes[0].invert_yaxis(); axes[0].set_xlabel('Clean balanced-accuracy difference (percentage points)')
    handles,labels = axes[0].get_legend_handles_labels()
    figure.legend(handles,labels,fontsize=8,loc='lower center',bbox_to_anchor=(.5,.06),ncol=3)
    axes[0].set_title('No recent-method predictive advantage established',fontsize=10)
    axes[1].barh(y,[100*e[1] for e in entries],color=['#899baa','#007b83','#ba5b36']); axes[1].set_yticks(y,[]); axes[1].invert_yaxis(); axes[1].set_xlim(0,15); axes[1].set_xlabel('Attributes removed (%)')
    for j,e in enumerate(entries): axes[1].text(100*e[1]+.25,j,f'{100*e[1]:.2f}%',va='center',fontsize=9)
    axes[1].set_title('Count changes must remain explicit',fontsize=10)
    figure.suptitle('Development evidence: all 15 datasets, independent audits',fontsize=12)
    figure.text(.01,.01,'Descriptive dataset-bootstrap intervals; repeatedly exposed development data. Sobolev optimizer flags retained.',fontsize=8)
    figure.tight_layout(rect=[0,.15,1,.96]); target = ROOT/'results/figures/method-development/latest-negative-evidence'; target.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(target.with_suffix('.png'),dpi=180); figure.savefig(target.with_suffix('.svg')); plt.close(figure)
    target.with_suffix('.json').write_text(json.dumps({'entries':entries,'scope':'not_confirmation'},indent=2),encoding='utf-8')
    lines = ['# Latest method/code experiments: candid review','',
        '2026-10-10. Goal ACTIVE / UNACHIEVED. No TeX work. All15 are repeatedly exposed development data.','',
        '## Outcome','',
        'Neither training-marginal product refinement nor Sobolev classifier correction establishes superior prediction against both recent same-count controls. Generic ANOVA, forest cardinality DP and Sobolev tools are not verified novelty. Repeated negative results do not establish that no better method exists, but further arbitrary grids on these same data are not confirmation.','',
        '|Experiment|Clean BA %|Removal %|Gain versus equal-count FGS pp|Gain versus equal-count SFRIP pp|Numeric gate|',
        '|---|---:|---:|---:|---:|---|',
        f"|Uniform ANOVA DP, count restoration|{100*dp['means']['dp_refit']['clean']:.4f}|{100*dp['means']['dp_refit']['removed']:.4f}|{dp['contrasts']['clean']['fgs_matched']['mean_pp']:.4f}|{dp['contrasts']['clean']['sfrip_matched']['mean_pp']:.4f}|FAIL|",
        f"|Empirical product, feasible fixed count|{100*empirical['means']['empirical_refit']['clean']:.4f}|{100*empirical['means']['empirical_refit']['removed']:.4f}|{empirical['contrasts']['clean']['fgs_matched']['mean_pp']:.4f}|{empirical['contrasts']['clean']['sfrip_matched']['mean_pp']:.4f}|FAIL|",
        f"|Sobolev, common training-only CV|{100*sobolev['means']['refined']['clean']:.4f}|{100*sobolev['means']['refined']['removed']:.4f}|{sobolev['contrasts']['clean']['fgs_matched']['mean_pp']:.4f}|{sobolev['contrasts']['clean']['sfrip_matched']['mean_pp']:.4f}|FAIL|",'',
        '## What changed in code','',
        'Reference-marginal ANOVA uses class-balanced training weights and handles degenerate coordinates without inversion. A fixed-count feasible DP/swap search retains the old feasible support as explicit fallback and never increases attributes to improve the comparison. Every accepted support descends the declared product reconstruction loss and passes the original FRS conditions. This product law is not the observed joint distribution.','',
        f"Empirical search changes {empirical['counts']['empirical_changed']}/128 supports; {empirical['counts']['empirical_rejections']}/{empirical['counts']['empirical_attempts']} attempted replacements reject. Causes: {dict(rejected)}. These are dependent search counts, not independent test units. Surrogate descent does not guarantee refitted-classifier accuracy.",'',
        'Sobolev hats whiten H+alpha*K and use the same classifier choices for every support method. Alpha0 ablations isolate shared smoothness correction; old-selector comparison isolates feature selection. Exact rational checks and ordinary numerical tests establish refinement identities, not population performance.','',
        '## Numerical limitations kept visible','',
        'Functional normalization experiment14 is COMPLETE but its full audit FAILED. Seven first-failure cases are archived; all have small objective gaps and no changed training predictions, but predeclared intercept tolerance fails. One occurs in a selected FGS child model. No tolerance was loosened and no accepted full-audit claim is made. Consolidated older script requiring this audit must not be run as if it had passed.','',
        f"Sobolev original outputs retain {sobolev['counts']['parent_caps']} parent and {sobolev['counts']['child_caps']} child stationarity failures; these counts include gradient-threshold failures, not necessarily iteration caps. The audit verifies that flags correctly describe the frozen fits. Numeric gate fails even apart from negative comparative effects.",'',
        f"Training-only Newton stationarity correction diagnoses {numeric['flagged_models']} flagged models, leaves {numeric['remaining_flags']} flagged, and changes {numeric['prediction_changes']} recorded validation/query predictions. Frozen fits, CV traces and scientific outputs are NOT overwritten. This is numerical code improvement, not new predictive evidence. Original v1 correction and diagnostic are archived separately.",'',
        '## Verification','',
        f"Empirical full audit: {empaudit['counts']}.",'',
        f"Sobolev full audit: {sobaudit['counts']}. Parameter/score/objective maxima: {sobaudit['precision_maxima']}.",'',
        '## Next research requirement','',
        'Do not expand either failed hypothesis to untouched confirmation or select favorable datasets. Before another predictive experiment, identify a genuinely different task-risk/conditional-distribution mechanism and validate its training-only behavior. Faithful FD+/FCE+ still needs unambiguous published operators or public author artifacts; existing proxies must remain labeled proxies. A high-level journal/SOTA claim also requires fully comparable recent baseline tuning, actual-count curves, novelty audit, and untouched confirmation.','',
        'All per-reference cases, including every adverse result, are in latest-method-development-review.json; scientific plot and source data are under results/figures/method-development/latest-negative-evidence.*.']
    (ROOT/'research/latest-method-development-review.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print({'goal_achieved':False,'empirical_rejection_causes':dict(rejected),'review':'research/latest-method-development-review.md'},flush=True)

if __name__ == '__main__': main()
