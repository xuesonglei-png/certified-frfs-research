"""Read-only mechanisms on independently audited exposed development rows."""
import argparse
import json
import time
import numpy as np
from run_fixed_face_frs_development_v2 import ROOT,FOLDER,ARMS,digest


def diagnose(name):
    path=FOLDER/(name+'.json');audit_path=FOLDER/(name+'.audit.json')
    audit=json.loads(audit_path.read_text());assert audit['passed'] and audit['sha256']==digest(path)
    saved=json.loads(path.read_text());out={'dataset':name,'source_sha256':digest(path),
        'audit_sha256':digest(audit_path),'observed_unix':time.time(),'exposed_development_only':True,
        'goal_achieved':False,'fixed_deletions':saved['fixed_zero_indices'],'arms':{}}
    for arm in ARMS:
        point=np.asarray(saved['fits'][arm]['target']) if saved['available'] else None
        fit=saved['fits'][arm]['fit'] if saved['available'] else None
        facial=arm.startswith('face_');controls=('joint_pair','full_support_'+arm.split('_')[1]) if facial else ('anchor',)
        report=saved['results'][arm]['evaluation'];comparisons={}
        for control in controls:
            previous=saved['results'][control]['evaluation']
            comparisons[control]={
                'neighbor_sets_changed_queries':sum(set(a)!=set(b) for a,b in zip(report['neighbor_indices'],previous['neighbor_indices'],strict=True)),
                'ordered_neighbor_lists_changed_queries':sum(a!=b for a,b in zip(report['neighbor_indices'],previous['neighbor_indices'],strict=True)),
                'predictions_changed_queries':sum(a!=b for a,b in zip(report['predictions'],previous['predictions'],strict=True)),
                'clean_BA_delta_pp':100*(report['metrics']['clean_BA']-previous['metrics']['clean_BA']),
                'perturbed_BA_delta_pp':100*(report['metrics']['mean_perturbed_BA']-previous['metrics']['mean_perturbed_BA'])}
        out['arms'][arm]={'comparisons':comparisons,'query_count':len(report['predictions']),
            'training_objective_descent_from_actual_start':None if fit is None else fit['optimization_start_objective']-fit['training_objective_final'],
            'normalized_weight_L1_movement_from_actual_start':None if fit is None else float(abs(point-np.asarray(fit['optimization_start_weights'])).sum()/fit['mass']),
            'fixed_zeros_preserved':None if fit is None else bool(np.all(point[saved['fixed_zero_indices']]==0.)) if facial else None,
            'fit_reuse':saved['fit_reuse'].get(arm)}
    target=ROOT/'research'/('fixed-face-FRS-mechanism-'+name+'.json')
    assert not target.exists(), 'Retain old diagnosis; use a new version instead of overwrite'
    target.write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);args=p.parse_args();diagnose(args.dataset)
