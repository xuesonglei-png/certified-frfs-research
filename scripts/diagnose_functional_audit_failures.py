"""Locate first frozen tolerance violation per dataset; never certify a pass."""
import json
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_nested_joint_path import ROOT,digest
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from functional_spline_audit_oracle import refit_functional

FOLDER = ROOT/'results/functional-normalization-pilot'

def diagnose(name):
    threadpool_limits(1); data = json.loads((FOLDER/(name+'.json')).read_text())
    X,y = load_dataset(name); checked = 0
    for r in data['records']:
        ri,a = np.array(r['outer_reference_indices']),np.array(r['reference_indices'])
        raw,labels = X[ri[a]],y[ri[a]]
        for fi,fold in enumerate(r['tuning']['folds']):
            ca,cb = np.array(fold['reference_indices']),np.array(fold['validation_indices'])
            AA,_ = preprocess_fit(raw[ca],raw[cb]); train = labels[ca]
            for mid,model in enumerate(fold['models']):
                fit = model['fit']; diag = refit_functional(AA,train,fit,strict=False); checked += 1
                if 'comparisons' not in diag: continue
                if not all(diag['comparisons'].values()) or not diag['stationarity_passed'] or not diag['objective_passed']:
                    selected = [m for m,c in r['tuning']['choices'].items()
                        if r['tuning']['trace'][m][c['config_index']]['fold_model_ids'][fi] == mid]
                    answer = {'dataset':name,'outer':r['outer_index'],'inner':r['inner_index'],
                        'child':fi,'model':mid,'checked_models':checked,'selected_by':selected,
                        'C':fit['C'],'knots':len(fit['knots'][0]),'pair_penalty':fit['pair_penalty'],
                        'production_iterations':fit['optimizer_iterations'],'production_flag':fit['optimizer_converged'],
                        'diagnostics':diag,'artifact_sha256':digest(FOLDER/(name+'.json'))}
                    print(answer,flush=True); return answer
    return {'dataset':name,'checked_models':checked,'first_child_failure':None,
        'scope':'diagnosis_not_full_audit_parent_and_queries_not_checked'}

if __name__ == '__main__':
    # Previously missing audit checkpoints; all are retained regardless of result.
    names = ['ecoli','glass','heart_statlog','ionosphere','transfusion','vehicle','yeast']
    with ProcessPoolExecutor(max_workers=2) as pool:
        rows = [f.result() for f in as_completed([pool.submit(diagnose,n) for n in names])]
    result = {'audit_passed':False,'tolerances_unchanged':True,'rows':rows,
        'oracle_sha256':digest(ROOT/'scripts/functional_spline_audit_oracle.py')}
    (ROOT/'results/functional-audit-failure-diagnosis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(result,flush=True)
