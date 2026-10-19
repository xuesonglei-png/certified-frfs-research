"""After valid pilot audit, freeze/extend all15 inner folds and replay all."""
import json
import subprocess
import sys
import time
import traceback
from run_majority_hinge_all15_diagnostic import ROOT,FOLDER,SOURCE,NAMES,REFINED,digest
from continue_hardaware_soft5_knn_pilot import wait_verified

STATUS=ROOT/'results/majority-hinge-inner-all15-chain-status.json'
PILOT=ROOT/'results/majority-hinge-inner-pilot'


def status(phase,**values):
    STATUS.write_text(json.dumps({'phase':phase,'updated_unix':time.time(),'goal_achieved':False,**values},indent=2),encoding='utf-8')


def main(pid):
    status('waiting_for_pilot_v2_independent_audit',prerequisite_pid=pid)
    audit=wait_verified(PILOT,pid);summary=json.loads((PILOT/'complete.json').read_text())
    assert audit['counts']['datasets']==3 and audit['counts']['inner_folds']==9
    assert summary['target_optimizer_failures']==0
    repair=json.loads((ROOT/'research/majority-hinge-auditor-v2-repair.json').read_text())
    for rel,sha in repair['fingerprints'].items():assert digest(ROOT/rel)==sha,rel
    assert audit['auditor_sha256']==digest(ROOT/'scripts/audit_majority_hinge_inner_pilot_v2.py')
    pilot_review={'audit_sha256':digest(PILOT/'complete-audit.json'),'mean_inner_gain_pp':summary['mean_inner_gain_pp'],
        'original_audit_failure_preserved':True,'goal_achieved':False,'new_outer_evaluation_performed':False,
        'decision':'Extend fixed objective to all15 inner folds; no predictive superiority claim.'}
    (ROOT/'research/majority-hinge-inner-pilot-review.json').write_text(json.dumps(pilot_review,indent=2),encoding='utf-8')
    frozen_path=ROOT/'research/majority-hinge-inner-all15-source-freeze.json'
    if frozen_path.exists():freeze=json.loads(frozen_path.read_text())
    else:
        files=list((ROOT/'scripts').glob('*.py'))+list((ROOT/'src/certified_frfs').glob('*.py'))
        files+=[ROOT/'data/processed'/(n+'.npz') for n in NAMES]
        files+=[ROOT/'research/majority-hinge-inner-all15-protocol.md',ROOT/'tests/test_cached_hinge_cut_validation.py',
                SOURCE/'complete-audit.json',PILOT/'complete-audit.json',PILOT/'complete.json']
        files+=[PILOT/(n+'.json') for n in ('iris','heart_statlog','sonar')]
        freeze={'registered_unix':time.time(),'registered_before_new_scoring':True,
                'fingerprints':{p.relative_to(ROOT).as_posix():digest(p) for p in files},
                'scope':'all15_exposed_inner_only_fixed_convex_target; three_audited_pilot_cohorts_reused'}
        frozen_path.write_text(json.dumps(freeze,indent=2),encoding='utf-8')
    for rel,sha in freeze['fingerprints'].items():assert digest(ROOT/rel)==sha,rel
    FOLDER.mkdir(exist_ok=True)
    for n in ('iris','heart_statlog','sonar'):
        source_path=SOURCE/(n+'.json');path=FOLDER/(n+'.json')
        hashes={**freeze['fingerprints'],source_path.relative_to(ROOT).as_posix():digest(source_path)}
        if path.exists():assert json.loads(path.read_text())['fingerprints']==hashes;continue
        saved=json.loads((PILOT/(n+'.json')).read_text());assert audit['artifact_hashes'][n]==digest(PILOT/(n+'.json'))
        provenance={'pilot_sha256':audit['artifact_hashes'][n],'pilot_audit_sha256':digest(PILOT/'complete-audit.json'),
                    'original_fingerprints':saved['fingerprints'],'unchanged_fit_options_predictions_and_selections':True}
        saved.update({'fingerprints':hashes,'pilot_reuse_provenance':provenance,
                      'scope':'all15_exposed_inner_only_common_convex_majority_hinge_prototype'})
        path.write_text(json.dumps(saved,separators=(',',':')),encoding='utf-8')
    for phase,script in [('running','run_majority_hinge_all15_diagnostic.py'),('auditing','audit_majority_hinge_all15_diagnostic.py')]:
        log=ROOT/'results'/f'majority-hinge-inner-all15-{phase}.log'
        with log.open('a',encoding='utf-8') as stream:
            child=subprocess.Popen([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            status(phase,child_pid=child.pid,log=log.relative_to(ROOT).as_posix())
            if child.wait():raise RuntimeError(f'{script} failed')
    result=json.loads((FOLDER/'complete.json').read_text());independent=json.loads((FOLDER/'complete-audit.json').read_text())
    assert independent['passed'] and independent['summary_sha256']==digest(FOLDER/'complete.json')
    rows=[]
    for n in NAMES:
        saved=json.loads((FOLDER/(n+'.json')).read_text())
        rows.append({'dataset':n,'gain_pp':{m:saved['selection'][m]['selected_inner_gain_pp'] for m in REFINED},
            'selected':{m:saved['selection'][m]['selected'] for m in REFINED}})
    review={'summary_sha256':digest(FOLDER/'complete.json'),'audit_sha256':digest(FOLDER/'complete-audit.json'),
        'mean_inner_gain_pp':result['mean_inner_gain_pp'],'rows':rows,'target_optimizer_failures':result['target_optimizer_failures'],
        'three_pilot_cohorts_reused':True,'all44_folds_reaudited':True,'new_outer_evaluation_performed':False,'goal_achieved':False,
        'scope':'all15_inner_validation_extension_not_classification_superiority_or_submission_readiness',
        'limits':['All15 exposed development.','Anchor inclusion makes selected-inner nonworsening tautological.',
                 'Weighted native/countFGS120 comparison and fresh confirmation remain required.','Finite-table bound is not population generalization.']}
    (ROOT/'research/majority-hinge-inner-all15-review.json').write_text(json.dumps(review,indent=2),encoding='utf-8')
    status('complete_audited_reviewed',mean_inner_gain_pp=result['mean_inner_gain_pp'],target_optimizer_failures=result['target_optimizer_failures'])


if __name__=='__main__':
    try:main(int(sys.argv[1]))
    except Exception as error:status('failed',error=str(error));traceback.print_exc();raise
