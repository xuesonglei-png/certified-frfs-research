"""Recover strict rank-tie audit; reuse frozen raw fits, preserve v1 failure."""
import argparse
import ctypes
import json
import subprocess
import sys
import time
import traceback
from run_direct_majority_frs_development import ROOT,FOLDER,FREEZE,prerequisites,frozen,digest

STATUS=ROOT/'results/direct-majority-rank-recovery-v2-chain-status.json'
RECOVERY=ROOT/'research/direct-majority-rank-recovery-v2-source-freeze.json'
TESTS=ROOT/'research/direct-majority-rank-recovery-v2-tests.json'


def status(phase,**kw):
    STATUS.write_text(json.dumps({'phase':phase,'updated_unix':time.time(),'goal_achieved':False,
        'classification_superiority_established':False,**kw},indent=2),encoding='utf-8')


def unchanged():
    frozen()
    hashes=json.loads(RECOVERY.read_text())['fingerprints']
    for rel,sha in hashes.items():assert digest(ROOT/rel)==sha,rel


def execute(phase,script,args=()):
    unchanged();log=ROOT/'results'/('direct-majority-rank-recovery-v2-'+phase+'.log')
    with log.open('a',encoding='utf-8') as stream:
        child=subprocess.Popen([sys.executable,str(ROOT/'scripts'/script),*args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        status(phase,child_pid=child.pid,args=args,log=log.relative_to(ROOT).as_posix())
        if child.wait():raise RuntimeError(f'{script} failed; all original and recovery artifacts retained')


def wait_audit(pid):
    ready=FOLDER/'transfusion.audit-v2.json'
    if ready.exists():return
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong];kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)];kernel.GetExitCodeProcess.restype=ctypes.c_int
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=kernel.OpenProcess(0x1000,False,pid)
    if not handle:raise RuntimeError('Named smoke-audit process absent and no successful audit; do not duplicate')
    status('waiting_for_named_transfusion_v2_audit',verified_audit_pid=pid)
    try:
        while not ready.exists():
            code=ctypes.c_ulong();assert kernel.GetExitCodeProcess(handle,ctypes.byref(code))
            if code.value!=259:raise RuntimeError('Transfusion v2 smoke audit ended without success evidence')
            time.sleep(10)
    finally:kernel.CloseHandle(handle)


def main(pid):
    assert not RECOVERY.exists() and not (FOLDER/'complete-audit.json').exists()
    tests=json.loads(TESTS.read_text());assert tests['passed'] and tests['tests_passed']==2
    for rel,sha in tests['fingerprints'].items():assert digest(ROOT/rel)==sha,rel
    original=json.loads((ROOT/'results/direct-majority-frs-development-chain-status.json').read_text())
    assert original['phase']=='failed' and 'audit_' in original['error'];parents=prerequisites();frozen()
    paths=[p for n in ('src','scripts','tests') for p in (ROOT/n).rglob('*.py')]
    paths += [TESTS,FREEZE,ROOT/'research/direct-majority-rank-replay-v1-diagnosis.json',ROOT/'research/direct-majority-rank-recovery-v2-protocol.md']
    RECOVERY.write_text(json.dumps({'created_unix':time.time(),'fingerprints':{p.relative_to(ROOT).as_posix():digest(p) for p in paths},
        'original_failed_status':original,'scope':'auditor_only_nearest_witness_identity_ties_raw_model_and_predictions_unchanged'},indent=2),encoding='utf-8')
    wait_audit(pid)
    schedule=('iris','wine','seeds','transfusion','wdbc','vehicle','vertebral','heart_statlog','ionosphere','sonar','ecoli','glass','haberman','banknote','yeast')
    assert set(schedule)==set(parents['frs-joint-zero-face']['artifact_hashes'])
    auditor=ROOT/'scripts/audit_direct_majority_frs_development_v2.py'
    for name in schedule:
        execute('fitting_scoring','run_direct_majority_frs_development.py',('--dataset',name))
        ready=FOLDER/(name+'.audit-v2.json')
        if ready.exists():
            r=json.loads(ready.read_text());assert r['passed'] and r['sha256']==digest(FOLDER/(name+'.json')) and r['auditor_sha256']==digest(auditor)
        else:execute('dataset_auditing','audit_direct_majority_frs_development_v2.py',('--dataset',name))
    execute('summary','run_direct_majority_frs_development.py')
    execute('complete_audit','audit_direct_majority_frs_development_v2.py')
    summary=json.loads((FOLDER/'complete.json').read_text());audit=json.loads((FOLDER/'complete-audit.json').read_text())
    assert audit['passed'] and audit['summary_sha256']==digest(FOLDER/'complete.json')
    review={'audit_sha256':digest(FOLDER/'complete-audit.json'),'all_dataset_rows':summary['rows'],'means':summary['means'],
        'contrasts':summary['contrasts'],'mean_actual_feature_counts':summary['mean_actual_feature_counts'],'audit_counts':audit['counts'],
        'recovery_source_freeze_sha256':digest(RECOVERY),'original_failed_audit_retained':True,'classification_superiority_established':False,
        'goal_achieved':False,'limits':summary['limits'],'decision':'Review full matched predictive outcomes; recovery fixes tie identity only, never objective/certification.'}
    (ROOT/'research/direct-majority-FRS-development-review.json').write_text(json.dumps(review,indent=2),encoding='utf-8')
    status('complete_audited_reviewed',audit_sha256=review['audit_sha256'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prior-audit-pid',required=True,type=int);args=p.parse_args()
    try:main(args.prior_audit_pid)
    except Exception as e:status('failed',error=str(e));traceback.print_exc();raise
