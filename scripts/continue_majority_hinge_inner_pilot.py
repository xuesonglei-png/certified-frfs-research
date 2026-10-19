"""Execute and independently audit the registered convex-target pilot."""
import json
import subprocess
import sys
import time
import traceback
from run_majority_hinge_inner_pilot import ROOT,FOLDER,REFINED,NAMES,digest

STATUS=ROOT/'results/majority-hinge-inner-pilot-chain-status.json'


def status(phase,**values):
    STATUS.write_text(json.dumps({'phase':phase,'updated_unix':time.time(),'goal_achieved':False,**values},indent=2),encoding='utf-8')


def main():
    for phase,script in [('running','run_majority_hinge_inner_pilot.py'),('auditing','audit_majority_hinge_inner_pilot.py')]:
        log=ROOT/'results'/f'majority-hinge-inner-pilot-{phase}.log'
        with log.open('a',encoding='utf-8') as stream:
            child=subprocess.Popen([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            status(phase,child_pid=child.pid,log=log.relative_to(ROOT).as_posix())
            if child.wait():raise RuntimeError(f'{script} failed')
    summary=json.loads((FOLDER/'complete.json').read_text());audit=json.loads((FOLDER/'complete-audit.json').read_text())
    assert audit['passed'] and audit['summary_sha256']==digest(FOLDER/'complete.json')
    rows=[]
    for name in NAMES:
        source=json.loads((FOLDER/(name+'.json')).read_text());assert audit['artifact_hashes'][name]==digest(FOLDER/(name+'.json'))
        rows.append({'dataset':name,'selected':{m:source['selection'][m]['selected'] for m in REFINED},
            'gain_pp':{m:source['selection'][m]['selected_inner_gain_pp'] for m in REFINED}})
    review={'summary_sha256':digest(FOLDER/'complete.json'),'audit_sha256':digest(FOLDER/'complete-audit.json'),
        'rows':rows,'mean_inner_gain_pp':summary['mean_inner_gain_pp'],'goal_achieved':False,'new_outer_evaluation_performed':False,
        'prediction_superiority_established':False,'scope':'inner_only_common_majority_hinge_prototype',
        'limits':['Selection includes anchor; inner nonworsening is tautological.','Small three-dataset exposed pilot, not representative or confirmation.',
                 'Finite-table convex surrogate bound is not a population risk bound or proof of novel FRS theory.',
                 'Original certification thresholds unchanged; only equally improved controls can support a method gain.']}
    (ROOT/'research/majority-hinge-inner-pilot-review.json').write_text(json.dumps(review,indent=2),encoding='utf-8')
    status('complete_audited_reviewed',mean_inner_gain_pp=summary['mean_inner_gain_pp'],target_optimizer_failures=summary['target_optimizer_failures'])


if __name__=='__main__':
    try:main()
    except Exception as error:status('failed',error=str(error));traceback.print_exc();raise
