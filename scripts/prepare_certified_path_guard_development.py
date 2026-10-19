"""Prospective freeze, smoke, complete ALL15 cohort and independent review."""
import json
import subprocess
import sys
import time
import traceback
from run_certified_path_guard_development import ROOT,PARENTS,FOLDER,FREEZE,prerequisites,frozen,digest

STATUS=ROOT/'results/certified-path-guard-development-chain-status.json'
TESTS=ROOT/'research/certified-path-guard-development-tests.json'


def status(phase,**kw):
    STATUS.write_text(json.dumps({'phase':phase,'updated_unix':time.time(),'goal_achieved':False,
        'classification_superiority_established':False,'new_outer_evaluation_performed':False,**kw},indent=2),encoding='utf-8')


def execute(phase,script,args=()):
    frozen();log=ROOT/'results'/('certified-path-guard-development-'+phase+'.log')
    with log.open('a',encoding='utf-8') as stream:
        child=subprocess.Popen([sys.executable,str(ROOT/'scripts'/script),*args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        status(phase,child_pid=child.pid,log=log.relative_to(ROOT).as_posix(),args=args)
        if child.wait():raise RuntimeError(f'{script} failed; artifacts retained')


def main():
    assert not FREEZE.exists() and not FOLDER.exists()
    tests=json.loads(TESTS.read_text());assert tests['passed'] and tests['tests_passed']==2
    for rel,sha in tests['fingerprints'].items():assert digest(ROOT/rel)==sha,rel
    parents=prerequisites();status('freezing')
    paths=[p for n in ('src','scripts','tests') for p in (ROOT/n).rglob('*.py')]
    paths += [TESTS,ROOT/'research/certified-path-guard-development-protocol.md']
    for n,p in PARENTS.items():paths += [p/'complete.json',p/'complete-audit.json']+[p/(name+'.json') for name in sorted(parents[n]['artifact_hashes'])]
    FREEZE.write_text(json.dumps({'created_unix':time.time(),'fingerprints':{p.relative_to(ROOT).as_posix():digest(p) for p in paths},
        'scope':'ALL15_training_class_guard_complete_certified_path_fullnative_development','goal_achieved':False},indent=2),encoding='utf-8')
    execute('smoke_fitting_scoring','run_certified_path_guard_development.py',('--dataset','iris'))
    execute('smoke_auditing','audit_certified_path_guard_development.py',('--dataset','iris'))
    for name in sorted(parents['frs-joint-zero-face']['artifact_hashes']):
        if name=='iris':continue
        execute('fitting_scoring','run_certified_path_guard_development.py',('--dataset',name))
        execute('dataset_auditing','audit_certified_path_guard_development.py',('--dataset',name))
    execute('summary','run_certified_path_guard_development.py')
    execute('complete_audit','audit_certified_path_guard_development.py')
    summary=json.loads((FOLDER/'complete.json').read_text());audit=json.loads((FOLDER/'complete-audit.json').read_text())
    assert audit['passed'] and audit['summary_sha256']==digest(FOLDER/'complete.json')
    review={'audit_sha256':digest(FOLDER/'complete-audit.json'),'all_dataset_rows':summary['rows'],'means':summary['means'],
        'contrasts':summary['contrasts'],'mean_actual_feature_counts':summary['mean_actual_feature_counts'],
        'audit_counts':audit['counts'],'goal_achieved':False,'classification_superiority_established':False,
        'decision':'Assess all15 recovery and compactness vs strong controls; no automatic promotion or significance.',
        'limits':summary['limits']}
    (ROOT/'research/certified-path-guard-development-review.json').write_text(json.dumps(review,indent=2),encoding='utf-8')
    status('complete_audited_reviewed',audit_sha256=review['audit_sha256'])


if __name__=='__main__':
    try:main()
    except Exception as e:status('failed',error=str(e));traceback.print_exc();raise
