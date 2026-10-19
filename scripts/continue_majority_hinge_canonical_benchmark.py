"""Registered full controls; run/audit/review without promoting success."""
import json
import subprocess
import sys
import time
import traceback
import numpy as np
from run_majority_hinge_canonical_benchmark import ROOT,FOLDER,PRIMARY,digest

STATUS=ROOT/'results/majority-hinge-canonical-benchmark-chain-status.json'


def status(phase,**values):
    STATUS.write_text(json.dumps({'phase':phase,'updated_unix':time.time(),'goal_achieved':False,**values},indent=2),encoding='utf-8')


def main():
    for phase,script in [('running','run_majority_hinge_canonical_benchmark.py'),('auditing','audit_majority_hinge_canonical_benchmark.py')]:
        log=ROOT/'results'/f'majority-hinge-canonical-benchmark-{phase}.log'
        with log.open('a',encoding='utf-8') as stream:
            child=subprocess.Popen([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            status(phase,child_pid=child.pid,log=log.relative_to(ROOT).as_posix())
            if child.wait():raise RuntimeError(f'{script} failed')
    summary=json.loads((FOLDER/'complete.json').read_text());audit=json.loads((FOLDER/'complete-audit.json').read_text())
    assert audit['passed'] and audit['summary_sha256']==digest(FOLDER/'complete.json')
    states={};constraints=0;vacuous=[]
    for r in summary['rows']:
        name=r['dataset'];saved=json.loads((FOLDER/(name+'.json')).read_text());assert audit['artifact_hashes'][name]==digest(FOLDER/(name+'.json'))
        for b in saved['inner']+[saved['outer']]:
            for l in b['stage']['learners']:states[l['fit']['status']]=states.get(l['fit']['status'],0)+1
            op=saved['selection']['proposed']['selected_index']
            metric=b['options']['proposed'][op]['metric'] if 'options' in b else b['results']['proposed']['metric']
            guard=b['stage']['certificate_checks'][metric['deployed_guard_id']]
            constraints+=len(guard['certificate']['required'])
            vacuous.append(sum(v==0 for v in guard['certificate']['required']))
    review={'audit_sha256':digest(FOLDER/'complete-audit.json'),'summary_sha256':digest(FOLDER/'complete.json'),
        'pilot_gate_passed':summary['pilot_gate_passed'],'primary_contrasts':{m:summary['contrasts_clean_BA'][m] for m in PRIMARY},
        'primary_descriptive_holm':summary['primary_descriptive_holm'],'means':summary['means'],
        'optimizer_states':states,'protected_class_conditions':constraints,'vacuous_retention_class_conditions':sum(vacuous),
        'goal_achieved':False,'classification_superiority_established':False,
        'decision':'Freeze candidate method for fresh independent confirmation.' if summary['pilot_gate_passed'] else
                   'Development gate failed; preserve all negative cases and review the method bottleneck.',
        'limits':['ALL15 repeatedly exposed; significance is descriptive.','Finite-table robust bound is not population risk.',
                 'Sequential frozen-support refinement is not fresh joint nested selection.','Inherited failures and numerical baseline conventions remain.',
                 'Convex hinge/metric learning has prior art; extra FRS novelty must be established.']}
    (ROOT/'research/majority-hinge-canonical-benchmark-review.json').write_text(json.dumps(review,indent=2),encoding='utf-8')
    status('complete_audited_reviewed',pilot_gate_passed=summary['pilot_gate_passed'],
           audit_sha256=digest(FOLDER/'complete-audit.json'))


if __name__=='__main__':
    try:main()
    except Exception as error:status('failed',error=str(error));traceback.print_exc();raise
