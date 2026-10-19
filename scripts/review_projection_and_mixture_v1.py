"""Complete-audit-only review with fixed comparison identity and negative results."""
import json
from pathlib import Path
import numpy as np
from run_mixture_enemy_diagnostic_v1 import ROOT,digest


def audited(folder):
    s=json.loads((folder/'complete.json').read_text());a=json.loads((folder/'complete-audit.json').read_text())
    assert a['passed'] and a['summary_sha256']==digest(folder/'complete.json')
    for n,sha in a['artifact_hashes'].items():assert digest(folder/(n+'.json'))==sha
    return s,a


def main():
    projection=ROOT/'results/robustaware-projection-v1';mixture=ROOT/'results/mixture-dependency-development-v1'
    s,a=audited(projection);m,ma=audited(mixture);assert s['datasets']==m['datasets']==15
    rows=[]
    for r in s['rows']:
        b,n=r['baseline_metrics'],r['proposed_metrics']
        rows.append({'dataset':r['dataset'],'baseline_BA':b['clean_BA'],'proposed_BA':n['clean_BA'],
            'BA_delta_pp':100*(n['clean_BA']-b['clean_BA']),
            'perturbed_BA_delta_pp':100*(n['mean_perturbed_BA']-b['mean_perturbed_BA']),
            'query_certified_BA_delta_pp':100*(n['query_certified_BA']-b['query_certified_BA']),
            'baseline_original_joint_feasible':r['fit']['anchor_guard']['accepted'],
            'support_size':len(r['features']),'selected_interpolation':r['fit']['selected_interpolation']})
    result={'rows':rows,'datasets':15,'mean_baseline_BA':float(np.mean([r['baseline_BA'] for r in rows])),
        'mean_proposed_BA':float(np.mean([r['proposed_BA'] for r in rows])),
        'mean_BA_delta_pp':float(np.mean([r['BA_delta_pp'] for r in rows])),
        'wins':sum(r['BA_delta_pp']>1e-10 for r in rows),'losses':sum(r['BA_delta_pp']< -1e-10 for r in rows),
        'ties':sum(abs(r['BA_delta_pp'])<=1e-10 for r in rows),
        'original_joint_certified_anchors':sum(r['baseline_original_joint_feasible'] for r in rows),
        'mixture_mean_global_tightening':m['mean_global_tightening'],
        'mixture_original_joint_feasible':m['old_joint_feasible'],'mixture_new_joint_feasible':m['new_joint_feasible'],
        'projection_audit_sha256':digest(projection/'complete-audit.json'),
        'mixture_audit_sha256':digest(mixture/'complete-audit.json'),
        'projection_comparator':'audited_weighted_FGS120_atomic_exact_count_not_complete_native_FGS1680',
        'projection_is_adopted':False,'mixture_prediction_advantage_established':False,'goal_achieved':False,
        'decision':'Reject this robust-training projection; mixture theory tightens bounds but does not expand joint feasibility here.',
        'limits':['All15 are development-exposed.','Projection preserves support: no additional feature reduction.',
            'Ten anchors are outside original joint certificate; their unchanged predictions do not constitute a certified proposed method.',
            'Mixture witness dominance is mathematical; literature novelty and predictive value remain unestablished.']}
    (ROOT/'research/projection-and-mixture-v1-review.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
