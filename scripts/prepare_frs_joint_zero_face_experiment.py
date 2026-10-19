"""Wait for complete zero repair, freeze and execute matched joint-face study."""
import json
import subprocess
import sys
import time
import traceback
from run_frs_joint_zero_face_experiment import ROOT,SOURCE,FOLDER,FREEZE,prerequisites,frozen,digest
from continue_soft_topk_knn_pilot import verify

STATUS=ROOT/'results/frs-joint-zero-face-chain-status.json'
TESTS=ROOT/'research/frs-joint-zero-face-tests.json'


def status(phase,**values):
    STATUS.write_text(json.dumps({'phase':phase,'updated_unix':time.time(),'goal_achieved':False,
        'classification_scored':False,**values},indent=2),encoding='utf-8')


def execute(phase,script,args=()):
    frozen();log=ROOT/'results'/f'frs-joint-zero-face-{phase}.log'
    with log.open('a',encoding='utf-8') as stream:
        child=subprocess.Popen([sys.executable,str(ROOT/'scripts'/script),*args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        status(phase,child_pid=child.pid,script_sha256=digest(ROOT/'scripts'/script))
        if child.wait():
            raise RuntimeError(f'{script} failed; preserve artifacts and create a new version for any fix')


def main():
    assert not FREEZE.exists() and not FOLDER.exists()
    tests=json.loads(TESTS.read_text());assert tests['passed'] and tests['tests_passed']==2
    for rel,sha in tests['fingerprints'].items():
        assert digest(ROOT/rel)==sha,rel
    status('waiting_for_complete_zero_face_repair_audit')
    review_path=ROOT/'research/FRS-zero-face-repair-review.json'
    while not review_path.exists():
        p=ROOT/'results/frs-zero-face-repair-chain-status.json'
        if p.exists() and json.loads(p.read_text())['phase']=='failed':
            raise RuntimeError('Prior zero repair failed; no new joint study started')
        time.sleep(10)
    prerequisites();verify(ROOT/'results/frs-zero-face-repair')
    base=json.loads((ROOT/'research/frs-zero-face-repair-source-freeze.json').read_text())['fingerprints']
    for rel,sha in base.items():
        assert digest(ROOT/rel)==sha,rel
    paths=[p for folder in ('src','scripts','tests') for p in (ROOT/folder).rglob('*.py')]
    paths+=[TESTS,ROOT/'research/FRS-joint-zero-face-protocol.md',review_path,
        ROOT/'results/frs-zero-face-repair/complete.json',ROOT/'results/frs-zero-face-repair/complete-audit.json']
    hashes={**base,**{p.relative_to(ROOT).as_posix():digest(p) for p in paths}}
    FREEZE.write_text(json.dumps({'created_unix':time.time(),'fingerprints':hashes,
        'scope':'ALL15firsttrainingfoldmatched4regimesgreedyjointzerofaces','goal_achieved':False},indent=2),encoding='utf-8')
    execute('smoke_solving','run_frs_joint_zero_face_experiment.py',('--dataset','iris'))
    execute('smoke_auditing','audit_frs_joint_zero_face_experiment.py',('--dataset','iris'))
    smoke=json.loads((FOLDER/'iris.smoke-audit.json').read_text());assert smoke['passed'] and smoke['sha256']==digest(FOLDER/'iris.json')
    execute('solving','run_frs_joint_zero_face_experiment.py')
    execute('auditing','audit_frs_joint_zero_face_experiment.py')
    audit=verify(FOLDER);summary=json.loads((FOLDER/'complete.json').read_text())
    review={'audit_sha256':digest(FOLDER/'complete-audit.json'),'all_dataset_rows':summary['rows'],
        'counts':summary['counts'],'means':summary['means'],'audit_counts':audit['counts'],
        'goal_achieved':False,'classification_scored':False,'limits':summary['limits'],
        'decision':'Assess real common-vector certified sparsity before designing a separate matched predictive experiment.'}
    (ROOT/'research/FRS-joint-zero-face-review.json').write_text(json.dumps(review,indent=2),encoding='utf-8')
    status('complete_audited_reviewed',audit_sha256=review['audit_sha256'])


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        status('failed',error=str(error));traceback.print_exc();raise
