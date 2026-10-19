"""After ALL15 target audit, diagnose tighter FRS class certificates only."""
import json
import subprocess
import sys
import time
import traceback
from run_simplex_soft5_knn_pilot import ROOT,digest
from continue_hardaware_soft5_knn_pilot import wait_verified
from diagnose_classwise_star_admission import SOURCE,FOLDER

STATUS=ROOT/'results/classwise-star-admission-chain-status.json'


def status(phase,**values):
    STATUS.write_text(json.dumps({'phase':phase,'updated_unix':time.time(),'goal_achieved':False,**values},indent=2),encoding='utf-8')


def main(pid):
    status('waiting_for_all15_inner_audit',prerequisite_pid=pid)
    wait_verified(SOURCE,pid)
    for phase,script in [('running','diagnose_classwise_star_admission.py'),('auditing','audit_classwise_star_admission.py')]:
        log=ROOT/'results'/f'classwise-star-admission-{phase}.log'
        with log.open('a',encoding='utf-8') as stream:
            child=subprocess.Popen([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            status(phase,child_pid=child.pid,log=log.relative_to(ROOT).as_posix())
            if child.wait():raise RuntimeError(f'{script} failed')
    summary=json.loads((FOLDER/'complete.json').read_text());audit=json.loads((FOLDER/'complete-audit.json').read_text())
    assert audit['passed'] and audit['summary_sha256']==digest(FOLDER/'complete.json')
    review={'counts':summary['counts'],'audit_sha256':digest(FOLDER/'complete-audit.json'),
        'goal_achieved':False,'classification_scored':False,'scope':'classwise_FRS_tighter_certificate_not_prediction_advantage',
        'decision':'Measure the equally applied upgraded certificate in a registered classifier study.' if summary['counts']['new_admissions'] else
                   'No new feasible metric observed. Do not promote the symbolic example to an experimental classifier advantage.',
        'limits':['Ordinary float inflation, not directed rounding.','Finite class-prior robustness is not population risk.',
                 'Exactness requires stable common enemies and full/reduced unsaturation.','Novelty relative to prior classwise and signed-forest FRS work remains to be established.']}
    (ROOT/'research/classwise-star-admission-review.json').write_text(json.dumps(review,indent=2),encoding='utf-8')
    status('complete_audited_reviewed',counts=summary['counts'])


if __name__=='__main__':
    try:main(int(sys.argv[1]))
    except Exception as error:status('failed',error=str(error));traceback.print_exc();raise
