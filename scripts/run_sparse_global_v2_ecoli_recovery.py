"""Refit all three Ecoli sparse-arm starts with unchanged settings and dual guard."""
import json
import time
from pathlib import Path
from collections import Counter
import numpy as np
from threadpoolctl import threadpool_limits
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from fit_sparse_global_transition_frs_v2 import fit_sparse_global_transition_target
from audit_sparse_global_transition_frs_v2 import check_sparse_global_transition_fit
from run_mixture_enemy_diagnostic_v1 import ROOT,digest

OUT=ROOT/'results/sparse-global-v2-ecoli-recovery'
FREEZE=ROOT/'research/sparse-global-v2-ecoli-recovery-source-freeze.json'


def main():
    threadpool_limits(1);OUT.mkdir(exist_ok=True)
    source=ROOT/'results/sparse-global-frs-training/ecoli.json';old=json.loads(source.read_text())
    files=[Path(__file__),ROOT/'scripts/verified_global_cell_v2.py',ROOT/'scripts/fit_sparse_global_transition_frs_v2.py',
        ROOT/'scripts/audit_sparse_global_transition_frs_v2.py',ROOT/'tests/test_verified_global_cell_v2.py',source]
    files+=sorted((ROOT/'src/certified_frfs').glob('*.py'))
    frozen={p.relative_to(ROOT).as_posix():digest(p) for p in files}
    if not FREEZE.exists():FREEZE.write_text(json.dumps({'created_unix':time.time(),'fingerprints':frozen,
        'scope':'ALL_three_Ecoli_sparse_starts_dual_path_repair_same_settings_no_prediction'},indent=2),encoding='utf-8')
    assert json.loads(FREEZE.read_text())['fingerprints']==frozen
    X,y=load_dataset('ecoli');ri,ca=np.asarray(old['reference_indices']),np.asarray(old['training_indices'])
    A,_=preprocess_fit(X[ri][ca],X[ri][ca]);labels=y[ri][ca];block=old['block'];counts=Counter();rows=[]
    for start in ('anchor','common_capped','common_zero'):
        timer=time.perf_counter()
        target,fit=fit_sparse_global_transition_target(A,labels,block['features'],block['anchor'],
            full_reference=np.ones(A.shape[1])/np.sqrt(A.shape[1]),reference_scope='original_full',
            initialization=np.asarray(block['starts'][start]),initialization_scope=start)
        row={'features':block['features'],'anchor':block['anchor'],'target':target.tolist(),'fit':fit,
            'fit_seconds':time.perf_counter()-timer,'freeze_sha256':digest(FREEZE),'classification_scored':False}
        path=OUT/(start+'.json');path.write_text(json.dumps(row,separators=(',',':')),encoding='utf-8')
        check_sparse_global_transition_fit(A,labels,row,counts)
        rows.append({'start':start,'objective':fit['training_objective_final'],
            'accepted_steps':sum(s['accepted_descent'] for s in fit['steps']),
            'subproblem_failures':fit['subproblem_failures'],'sha256':digest(path)})
        print(start,'refit + independent fit/LP replay PASS',flush=True)
    report={'passed':True,'rows':rows,'counts':dict(counts),'freeze_sha256':digest(FREEZE),
        'source_sha256':digest(source),'classification_scored':False,'goal_achieved':False,
        'scope':'three_start_Ecoli_engineering_recovery_not_full_all15_ablation_or_predictive_gain'}
    (OUT/'complete-audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
