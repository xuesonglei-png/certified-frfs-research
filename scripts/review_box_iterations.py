"""Dataset-paired descriptive uncertainty; exposed data cannot confirm gains."""
import json
import numpy as np
from scipy.stats import wilcoxon
from run_decision_iteration import ROOT

def paired(values):
    x=np.array(values,float);rng=np.random.default_rng(20261010)
    boot=x[rng.integers(0,len(x),(20000,len(x)))].mean(axis=1)
    signs=1-2*((np.arange(2**len(x))[:,None]>>np.arange(len(x)))&1)
    perm=np.mean(signs*x,axis=1)
    return {'mean_pp':float(100*x.mean()),'bootstrap95_pp':(100*np.quantile(boot,[.025,.975])).tolist(),
        'sign_flip_two_sided_p':float(np.mean(abs(perm)>=abs(x.mean())-1e-14)),
        'wilcoxon_p':float(wilcoxon(x,zero_method='wilcox').pvalue) if np.any(abs(x)>1e-12) else 1.,
        'wins':int(sum(x>1e-12)),'losses':int(sum(x < -1e-12)),'ties':int(sum(abs(x)<=1e-12))}

def main():
    exact=json.loads((ROOT/'results/exact-box-matched-controls/complete.json').read_text())
    distilled=json.loads((ROOT/'results/distilled-box-matched-controls/complete.json').read_text())
    teacher=json.loads((ROOT/'results/teacher-support-refit/complete.json').read_text())
    matched={r['dataset']:r for r in distilled['rows']};rows=[]
    for row in teacher['rows']:
        name=row['dataset'];p=json.loads((ROOT/'results/teacher-support-refit'/f'{name}.json').read_text())
        trials=json.loads((ROOT/'results/distilled-box-matched-controls'/f'{name}.json').read_text())
        full=matched[name]['clean']['robust_full'];gains=[];noise=[]
        for s,c in zip(p['splits'],trials['splits']):
            assert s['reference_indices']==c['reference_indices'] and s['validation_indices']==c['validation_indices']
            gains.append(s['metrics']['clean']-c['results']['robust_full']['metrics']['clean'])
            noise.append(s['metrics']['query_noise01']-c['results']['robust_full']['metrics']['query_noise01'])
        rows.append({'dataset':name,'clean':row['clean'],'matched_full':full,'gain':float(np.mean(gains)),'noise_gain':float(np.mean(noise)),
            'removed':row['removed'],'new_feasible_candidates':sum(sum(c['certificate']['feasible'] for c in s['generated']) for s in p['splits']),
            'new_family_selected_splits':sum(s['selected']['candidate'].get('origin')=='reference_teacher_score_backward' for s in p['splits']),
            'full_support_splits':sum(len(s['selected']['candidate']['features'])==p['d'] for s in p['splits'])})
    result={'rows':rows,'exact_vs_matched_full':paired([r['gain_vs_robust_full'] for r in exact['rows']]),
        'distilled_vs_matched_full':paired([r['gain_vs_robust_full'] for r in distilled['rows']]),
        'teacher_support_vs_matched_full':paired([r['gain'] for r in rows]),
        'teacher_support_noise_vs_matched_full':paired([r['noise_gain'] for r in rows]),
        'teacher_support_vs_fgs_historical':paired([r['clean']-r['controls']['fgs2023'] for r in teacher['rows']]),
        'teacher_support_vs_sfrip_historical':paired([r['clean']-r['controls']['sfrip2025'] for r in teacher['rows']]),
        'scope':'dataset_equal_weight_descriptive_repeatedly_exposed_development_uncorrected_multiple_comparisons_not_confirmatory_significance'}
    (ROOT/'results/teacher-support-refit/complete-review.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print({k:v for k,v in result.items() if k!='rows'})

if __name__=='__main__':main()
