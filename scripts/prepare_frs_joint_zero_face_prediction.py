"""Wait for joint and canonical audit, freeze and run actual predictive screen."""
import json
import subprocess
import sys
import time
import traceback
from run_frs_joint_zero_face_prediction import ROOT,JOINT,NATIVE,FOLDER,FREEZE,prerequisites,frozen,digest
from continue_soft_topk_knn_pilot import verify

STATUS=ROOT/'results/frs-joint-zero-face-prediction-chain-status.json'
TESTS=ROOT/'research/frs-joint-zero-face-prediction-tests.json'


def status(phase,**values):
    STATUS.write_text(json.dumps({'phase':phase,'updated_unix':time.time(),'goal_achieved':False,
        'classification_superiority_established':False,'new_outer_evaluation_performed':False,**values},indent=2),encoding='utf-8')


def execute(phase,script,args=()):
    frozen();log=ROOT/'results'/f'frs-joint-zero-face-prediction-{phase}.log'
    with log.open('a',encoding='utf-8') as stream:
        child=subprocess.Popen([sys.executable,str(ROOT/'scripts'/script),*args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        status(phase,child_pid=child.pid,script_sha256=digest(ROOT/'scripts'/script))
        if child.wait():
            raise RuntimeError(f'{script} failed; preserve all artifacts')


def main():
    assert not FREEZE.exists() and not FOLDER.exists()
    tests=json.loads(TESTS.read_text());assert tests['passed'] and tests['tests_passed']==2
    for rel,sha in tests['fingerprints'].items():
        assert digest(ROOT/rel)==sha,rel
    status('waiting_for_full_joint_and_native_audits')
    review=ROOT/'research/FRS-joint-zero-face-review.json'
    while not (review.exists() and (NATIVE/'complete-audit.json').exists()):
        for name in ('frs-joint-zero-face-chain-status.json','majority-hinge-canonical-benchmark-chain-status.json'):
            path=ROOT/'results'/name
            if path.exists() and json.loads(path.read_text())['phase']=='failed':
                raise RuntimeError(f'{name} failed; no predictive scoring started')
        time.sleep(10)
    parents=prerequisites()
    base=json.loads((ROOT/'research/frs-joint-zero-face-source-freeze.json').read_text())['fingerprints']
    for rel,sha in base.items():
        assert digest(ROOT/rel)==sha,rel
    paths=[p for folder in ('src','scripts','tests') for p in (ROOT/folder).rglob('*.py')]
    paths+=[TESTS,ROOT/'research/FRS-joint-zero-face-prediction-protocol.md',review]
    for folder in (JOINT,NATIVE):
        paths+=[folder/'complete.json',folder/'complete-audit.json']
        paths+=[folder/(name+'.json') for name in sorted(parents[folder.name]['artifact_hashes'])]
    hashes={**base,**{p.relative_to(ROOT).as_posix():digest(p) for p in paths}}
    FREEZE.write_text(json.dumps({'created_unix':time.time(),'fingerprints':hashes,'goal_achieved':False,
        'scope':'all15_first_inner_joint_face_clean_query_and_16noise_fixed_full_native_controls'},indent=2),encoding='utf-8')
    execute('smoke_scoring','run_frs_joint_zero_face_prediction.py',('--dataset','iris'))
    execute('smoke_auditing','audit_frs_joint_zero_face_prediction.py',('--dataset','iris'))
    smoke=json.loads((FOLDER/'iris.smoke-audit.json').read_text());assert smoke['passed'] and smoke['sha256']==digest(FOLDER/'iris.json')
    execute('scoring','run_frs_joint_zero_face_prediction.py')
    execute('auditing','audit_frs_joint_zero_face_prediction.py')
    audit=verify(FOLDER);summary=json.loads((FOLDER/'complete.json').read_text())
    review={'audit_sha256':digest(FOLDER/'complete-audit.json'),'all_dataset_rows':summary['rows'],'means':summary['means'],
        'pair_clean_BA_differences_pp':summary['pair_clean_BA_differences_pp'],'pair_wins_losses':summary['pair_wins_losses'],
        'mean_actual_feature_counts':summary['mean_actual_feature_counts'],'audit_counts':audit['counts'],
        'classification_scored':True,'classification_superiority_established':False,'new_outer_evaluation_performed':False,
        'goal_achieved':False,'limits':summary['limits'],
        'decision':'Review all15 actual predictive and certified sparsity tradeoffs; no automatic promotion or significance claim.'}
    (ROOT/'research/FRS-joint-zero-face-prediction-review.json').write_text(json.dumps(review,indent=2),encoding='utf-8')
    status('complete_audited_reviewed',audit_sha256=review['audit_sha256'])


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        status('failed',error=str(error));traceback.print_exc();raise
