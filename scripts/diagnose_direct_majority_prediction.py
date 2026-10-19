"""Compare direct training improvement with actual independent query changes."""
import argparse
import json
import time
from run_direct_majority_frs_development import ROOT,FOLDER,ARMS,digest


def diagnose(name):
    path=FOLDER/(name+'.json');audit_path=FOLDER/(name+'.audit.json')
    audit=json.loads(audit_path.read_text());assert audit['passed'] and audit['sha256']==digest(path)
    saved=json.loads(path.read_text());out={'dataset':name,'source_sha256':digest(path),'audit_sha256':digest(audit_path),
        'observed_unix':time.time(),'exposed_development_only':True,'goal_achieved':False,'fixed_deletions':saved['fixed_zero_indices'],'arms':{}}
    for arm in ARMS:
        report=saved['results'][arm]['evaluation'];controls=('joint_pair','direct_control','fgs_weighted_native120') if arm=='direct_face' else ('anchor',)
        fit=saved['fits'].get(arm);comparisons={}
        for control in controls:
            previous=saved['results'][control]['evaluation']
            comparisons[control]={
                'neighbor_sets_changed_queries':sum(set(a)!=set(b) for a,b in zip(report['neighbor_indices'],previous['neighbor_indices'],strict=True)),
                'predictions_changed_queries':sum(a!=b for a,b in zip(report['predictions'],previous['predictions'],strict=True)),
                'clean_BA_delta_pp':100*(report['metrics']['clean_BA']-previous['metrics']['clean_BA']),
                'perturbed_BA_delta_pp':100*(report['metrics']['mean_perturbed_BA']-previous['metrics']['mean_perturbed_BA'])}
        out['arms'][arm]={'comparisons':comparisons,'query_count':len(report['predictions']),
            'training_objective_decrease':None if fit is None else fit['initial_objective']-fit['final_objective'],
            'training_qualified_rate_gain_pp':None if fit is None else 100*(fit['final_report']['balanced_qualified_rate']-fit['seed_report']['balanced_qualified_rate']),
            'mip_status':None if fit is None else fit['mip_status'],'accepted_descent':None if fit is None else fit['accepted_descent']}
    target=ROOT/'research'/('direct-majority-FRS-mechanism-'+name+'.json');assert not target.exists()
    target.write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);args=p.parse_args();diagnose(args.dataset)
