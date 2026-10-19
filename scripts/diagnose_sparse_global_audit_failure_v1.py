"""Inspect every sparse-arm trace flag without repairing or waiving it."""
import json
from pathlib import Path
import numpy as np
from independent_global_frs_transition import scalar_cell_evaluation

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/sparse-global-frs-training/ecoli.json'
OUT=ROOT/'research/sparse-global-audit-failure-v1.json'


def differences(a,b,path=''):
    out=[]
    if isinstance(a,dict):
        for k in a:out+=differences(a[k],b[k],path+'/'+k)
    elif isinstance(a,(list,tuple)):
        for i,(x,y) in enumerate(zip(a,b,strict=True)):out+=differences(x,y,path+'/'+str(i))
    elif isinstance(a,(bool,str)) or a is None:
        if a!=b:out.append({'field':path,'scalar':a,'stored':b})
    elif not np.allclose(a,b,atol=1e-10,rtol=1e-9):
        out.append({'field':path,'scalar':a,'stored':b,'difference':float(a)-float(b)})
    return out


def main():
    saved=json.loads(SOURCE.read_text());arm=saved['block']['arms']['global_signed_sparse'];rows=[]
    for start,r in arm['fits'].items():
        fit=r['fit']
        for sid,step in enumerate(fit['iterations'] if 'iterations' in fit else fit['steps']):
            if not step.get('last_master'):continue
            trace=step['subproblem_trace'][-1];actual=scalar_cell_evaluation(fit['class_cell'],step['candidate_weights'])
            delta=differences(actual,trace['class_cell_evaluation'])
            if delta:
                rows.append({'start':start,'step':sid,'subproblem_success':step['subproblem_success'],
                    'accepted_descent':step['accepted_descent'],'differences':delta,
                    'scalar_report':actual,'stored_report':trace['class_cell_evaluation'],
                    'candidate_equals_returned':bool(np.array_equal(step['candidate_weights'],r['target']))})
    out={'dataset':'ecoli','rows':rows,'mismatches':len(rows),
        'accepted_mismatches':sum(r['accepted_descent'] for r in rows),
        'successful_subproblem_mismatches':sum(r['subproblem_success'] for r in rows),
        'audit_rejection_waived':False,'classification_scored':False,'goal_achieved':False}
    OUT.write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))


if __name__=='__main__':main()
