"""Freeze, smoke, then ALL15 fitting/prediction/independent audit; no promotion."""
import json
import subprocess
import sys
import time
import traceback
from run_fixed_face_frs_development_v2 import ROOT,PARENTS,FOLDER,FREEZE,prerequisites,frozen,digest

STATUS=ROOT/'results/fixed-face-frs-development-v2-chain-status.json'
TESTS=ROOT/'research/fixed-face-FRS-development-v2-tests.json'


def status(phase,**kw):
    STATUS.write_text(json.dumps({'phase':phase,'updated_unix':time.time(),'goal_achieved':False,
        'classification_superiority_established':False,'new_outer_evaluation_performed':False,**kw},indent=2),encoding='utf-8')


def execute(phase,script,args=()):
    frozen();log=ROOT/'results'/('fixed-face-frs-development-v2-'+phase+'.log')
    with log.open('a',encoding='utf-8') as stream:
        child=subprocess.Popen([sys.executable,str(ROOT/'scripts'/script),*args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        status(phase,child_pid=child.pid,log=log.relative_to(ROOT).as_posix(),args=args)
        if child.wait():raise RuntimeError(f'{script} failed; artifacts retained, no silent fallback')


def main():
    assert not FREEZE.exists() and not FOLDER.exists()
    tests=json.loads(TESTS.read_text());assert tests['passed'] and tests['tests_passed']==3
    for rel,sha in tests['fingerprints'].items():assert digest(ROOT/rel)==sha,rel
    parents=prerequisites();status('freezing')
    paths=[p for n in ('src','scripts','tests') for p in (ROOT/n).rglob('*.py')]
    paths += [TESTS,ROOT/'research/fixed-face-FRS-development-v2-protocol.md',ROOT/'research/fixed-face-FRS-prototype-provenance.json']
    for n,p in PARENTS.items():
        paths += [p/'complete.json',p/'complete-audit.json']+[p/(name+'.json') for name in sorted(parents[n]['artifact_hashes'])]
    FREEZE.write_text(json.dumps({'created_unix':time.time(),'fingerprints':{p.relative_to(ROOT).as_posix():digest(p) for p in paths},
        'scope':'ALL15_fixed_face_matched_loss_native_fullcontrols_firstinner_development','goal_achieved':False},indent=2),encoding='utf-8')
    # Small Iris smoke runs before committing compute to all fifteen datasets.
    execute('smoke_fitting_scoring','run_fixed_face_frs_development_v2.py',('--dataset','iris'))
    execute('smoke_auditing','audit_fixed_face_frs_development_v2.py',('--dataset','iris'))
    # Predeclared schedule; every dataset is run, no result-based selection.
    schedule=('iris','wine','seeds','transfusion','wdbc','vehicle','vertebral','heart_statlog',
        'ionosphere','sonar','ecoli','glass','haberman','banknote','yeast')
    assert set(schedule)==set(parents['frs-joint-zero-face']['artifact_hashes'])
    for name in schedule:
        if name=='iris':continue
        execute('fitting_scoring','run_fixed_face_frs_development_v2.py',('--dataset',name))
        execute('dataset_auditing','audit_fixed_face_frs_development_v2.py',('--dataset',name))
    execute('summary','run_fixed_face_frs_development_v2.py')
    execute('complete_audit','audit_fixed_face_frs_development_v2.py')
    summary=json.loads((FOLDER/'complete.json').read_text());audit=json.loads((FOLDER/'complete-audit.json').read_text())
    assert audit['passed'] and audit['summary_sha256']==digest(FOLDER/'complete.json')
    review={'audit_sha256':digest(FOLDER/'complete-audit.json'),'all_dataset_rows':summary['rows'],
        'means':summary['means'],'contrasts':summary['contrasts'],'mean_actual_feature_counts':summary['mean_actual_feature_counts'],
        'audit_counts':audit['counts'],'goal_achieved':False,'classification_superiority_established':False,
        'decision':'Review both facial arms and strong native controls; retain failures, no accuracy or significance promise.',
        'limits':summary['limits']}
    (ROOT/'research/fixed-face-FRS-development-v2-review.json').write_text(json.dumps(review,indent=2),encoding='utf-8')
    status('complete_audited_reviewed',audit_sha256=review['audit_sha256'])


if __name__=='__main__':
    try:main()
    except Exception as e:
        status('failed',error=str(e));traceback.print_exc();raise
