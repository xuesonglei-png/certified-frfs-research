"""Prospective freeze then per-row audited singleton dependency and full15 run."""
import ctypes
import json
import subprocess
import sys
import time
import traceback
from run_ensemble_vote_certificate_development_v1 import ROOT,FOLDER,FREEZE,PARENTS,SINGLETON,SINGLETON_FREEZE,frozen,digest
from continue_soft_topk_knn_pilot import verify

STATUS=ROOT/'results/ensemble-vote-certificate-development-v1-chain-status.json'
TESTS=ROOT/'research/ensemble-vote-certificate-development-v1-tests.json'


def status(phase,**extra):
    STATUS.write_text(json.dumps({'phase':phase,'updated_unix':time.time(),
        'goal_achieved':False,'classification_superiority_established':False,
        'new_outer_evaluation_performed':False,**extra},indent=2),encoding='utf-8')


def available_physical_memory():
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(key,ctypes.c_ulonglong)
            for key in ('total_physical','available_physical','total_page','available_page',
                'total_virtual','available_virtual','available_extended')]
    memory=MEMORYSTATUSEX();memory.length=ctypes.sizeof(memory)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory))
    return memory.available_physical


def execute(phase,script,args=()):
    frozen()
    while available_physical_memory()<3*1024**3:
        status('waiting_memory',next_phase=phase,next_args=args,minimum_available_GiB=3)
        time.sleep(30.)
    log=ROOT/'results'/('ensemble-vote-certificate-development-v1-'+phase+'.log')
    with log.open('a',encoding='utf-8') as stream:
        child=subprocess.Popen([sys.executable,str(ROOT/'scripts'/script),*args],cwd=ROOT,
            stdout=stream,stderr=subprocess.STDOUT)
        status(phase,child_pid=child.pid,args=args,log=log.relative_to(ROOT).as_posix())
        if child.wait():raise RuntimeError(f'{script} failed; artifacts preserved')


def main():
    assert not FREEZE.exists() and not FOLDER.exists()
    tests=json.loads(TESTS.read_text());assert tests['passed'] and tests['tests_passed']==5
    for rel,sha in tests['fingerprints'].items():assert digest(ROOT/rel)==sha,rel
    status('verifying_complete_original_parent_audits')
    parents={n:verify(p) for n,p in PARENTS.items()}
    names=sorted(parents['frs-joint-zero-face']['artifact_hashes'])
    assert len(names)==15 and all(set(v['artifact_hashes'])==set(names) for v in parents.values())
    assert SINGLETON_FREEZE.exists()
    paths=[p for folder in ('src','scripts','tests') for p in (ROOT/folder).rglob('*.py')]
    paths += [TESTS,ROOT/'research/ensemble-vote-certificate-development-v1-protocol.md',SINGLETON_FREEZE]
    for n,p in PARENTS.items():paths += [p/'complete.json',p/'complete-audit.json']
    FREEZE.write_text(json.dumps({'created_unix':time.time(),
        'fingerprints':{p.relative_to(ROOT).as_posix():digest(p) for p in paths},
        'parents':parents,'singleton_freeze_sha256':digest(SINGLETON_FREEZE),'goal_achieved':False,
        'scope':'ALL15_8_arm_three_prespecified_partitions_same_original_certificates'},indent=2),encoding='utf-8')
    order=['iris','wine','ecoli','seeds','transfusion','wdbc','vehicle']
    order += [n for n in names if n not in order]
    for name in order:
        while not (SINGLETON/(name+'.audit.json')).exists():
            state=json.loads((ROOT/'results/plurality-certificate-development-v1-chain-status.json').read_text())
            if state['phase']=='failed':raise RuntimeError('singleton failed; new chain stopped without overriding failure')
            status('waiting_singleton_dataset_audit',dataset=name,prior_phase=state['phase'])
            time.sleep(30.)
        execute('fitting_scoring','run_ensemble_vote_certificate_development_v1.py',('--dataset',name))
        execute('dataset_auditing','audit_ensemble_vote_certificate_development_v1.py',('--dataset',name))
    # Each row was independently audited before reuse; the full singleton audit
    # is additionally mandatory before aggregate interpretation or promotion.
    while not (SINGLETON/'complete-audit.json').exists():
        state=json.loads((ROOT/'results/plurality-certificate-development-v1-chain-status.json').read_text())
        if state['phase']=='failed':raise RuntimeError('full singleton experiment failed; aggregate interpretation withheld')
        status('waiting_singleton_complete_audit');time.sleep(30.)
    verify(SINGLETON)
    execute('summary','run_ensemble_vote_certificate_development_v1.py')
    execute('complete_audit','audit_ensemble_vote_certificate_development_v1.py')
    summary=json.loads((FOLDER/'complete.json').read_text());audit=verify(FOLDER)
    review={**summary,'audit_sha256':digest(FOLDER/'complete-audit.json'),'audit_counts':audit['counts'],
        'singleton_complete_audit_sha256':digest(SINGLETON/'complete-audit.json'),
        'decision':'All15 development; matched singleton and strong native contrasts, no automatic promotion/significance.'}
    (ROOT/'research/ensemble-vote-certificate-development-v1-review.json').write_text(
        json.dumps(review,indent=2),encoding='utf-8')
    status('complete_audited_reviewed',audit_sha256=review['audit_sha256'])


if __name__=='__main__':
    try:main()
    except Exception as e:status('failed',error=str(e));traceback.print_exc();raise
