"""Freeze before actual fits; complete all15 with independent per-dataset audit."""
import json
import subprocess
import sys
import time
import traceback
from run_plurality_certificate_development_v1 import ROOT,FOLDER,FREEZE,PARENTS,frozen,digest
from continue_soft_topk_knn_pilot import verify

STATUS=ROOT/'results/plurality-certificate-development-v1-chain-status.json'
TESTS=ROOT/'research/plurality-certificate-development-v1-tests.json'


def status(phase,**extra):
    STATUS.write_text(json.dumps({'phase':phase,'updated_unix':time.time(),
        'goal_achieved':False,'classification_superiority_established':False,
        'new_outer_evaluation_performed':False,**extra},indent=2),encoding='utf-8')


def execute(phase,script,args=()):
    frozen();log=ROOT/'results'/('plurality-certificate-development-v1-'+phase+'.log')
    with log.open('a',encoding='utf-8') as stream:
        child=subprocess.Popen([sys.executable,str(ROOT/'scripts'/script),*args],cwd=ROOT,
            stdout=stream,stderr=subprocess.STDOUT)
        status(phase,child_pid=child.pid,args=args,log=log.relative_to(ROOT).as_posix())
        if child.wait():raise RuntimeError(f'{script} failed; artifacts preserved')


def main():
    assert not FREEZE.exists() and not FOLDER.exists()
    tests=json.loads(TESTS.read_text());assert tests['passed'] and tests['tests_passed']==6
    for rel,sha in tests['fingerprints'].items():assert digest(ROOT/rel)==sha,rel
    status('verifying_complete_parent_audits')
    parents={n:verify(p) for n,p in PARENTS.items()}
    names=sorted(parents['frs-joint-zero-face']['artifact_hashes'])
    assert len(names)==15 and all(set(v['artifact_hashes'])==set(names) for v in parents.values())
    paths=[p for folder in ('src','scripts','tests') for p in (ROOT/folder).rglob('*.py')]
    paths += [TESTS,ROOT/'research/plurality-certificate-development-v1-protocol.md']
    for n,p in PARENTS.items():paths += [p/'complete.json',p/'complete-audit.json']
    FREEZE.write_text(json.dumps({'created_unix':time.time(),
        'fingerprints':{p.relative_to(ROOT).as_posix():digest(p) for p in paths},
        'parents':parents,'goal_achieved':False,
        'scope':'ALL15_8_arm_calibrated_nominal_objective_original_reference_certificates'},indent=2),encoding='utf-8')
    order=['iris','wine','ecoli','seeds','transfusion','wdbc','vehicle']
    order += [n for n in names if n not in order]
    for name in order:
        execute('fitting_scoring','run_plurality_certificate_development_v1.py',('--dataset',name))
        execute('dataset_auditing','audit_plurality_certificate_development_v1.py',('--dataset',name))
    execute('summary','run_plurality_certificate_development_v1.py')
    execute('complete_audit','audit_plurality_certificate_development_v1.py')
    summary=json.loads((FOLDER/'complete.json').read_text());audit=verify(FOLDER)
    review={**summary,'audit_sha256':digest(FOLDER/'complete-audit.json'),'audit_counts':audit['counts'],
        'decision':'All15 development against matched and strong native controls; no automatic promotion/significance.'}
    (ROOT/'research/plurality-certificate-development-v1-review.json').write_text(
        json.dumps(review,indent=2),encoding='utf-8')
    status('complete_audited_reviewed',audit_sha256=review['audit_sha256'])


if __name__=='__main__':
    try:main()
    except Exception as e:status('failed',error=str(e));traceback.print_exc();raise
