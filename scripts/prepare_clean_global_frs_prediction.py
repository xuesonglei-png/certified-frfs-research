"""Await full training and native audits, freeze then actual ALL15 prediction."""
import json
import subprocess
import sys
import time
import traceback
from run_clean_global_frs_prediction import ROOT,SOURCE,NATIVE,FOLDER,FREEZE,REVIEW,prerequisites,frozen,digest
from continue_soft_topk_knn_pilot import verify

STATUS=ROOT/'results/clean-global-frs-prediction-chain-status.json'
TESTS=ROOT/'research/clean-global-FRS-prediction-tests.json'


def status(phase,**values):
    STATUS.write_text(json.dumps({'phase':phase,'updated_unix':time.time(),'goal_achieved':False,
        'new_outer_evaluation_performed':False,'classification_superiority_established':False,**values},indent=2),encoding='utf-8')


def execute(phase,script,args=()):
    frozen();log=ROOT/'results'/f'clean-global-frs-prediction-{phase}.log'
    with log.open('a',encoding='utf-8') as stream:
        child=subprocess.Popen([sys.executable,str(ROOT/'scripts'/script),*args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        status(phase,child_pid=child.pid,script_sha256=digest(ROOT/'scripts'/script))
        if child.wait():
            raise RuntimeError(f'{script} failed; retain artifacts and logs')


def main():
    assert not FREEZE.exists() and not FOLDER.exists()
    tests=json.loads(TESTS.read_text());assert tests['passed'] and tests['tests_passed']==2
    for rel,sha in tests['fingerprints'].items():
        assert digest(ROOT/rel)==sha,rel
    status('waiting_for_full_clean_training_and_native_audits')
    while not (REVIEW.exists() and (NATIVE/'complete-audit.json').exists()):
        for name in ('clean-global-frs-training-chain-status.json','majority-hinge-canonical-benchmark-chain-status.json'):
            p=ROOT/'results'/name
            if p.exists() and json.loads(p.read_text())['phase']=='failed':
                raise RuntimeError(f'{name} failed; no predictive scoring started')
        time.sleep(10)
    parents=prerequisites();review=json.loads(REVIEW.read_text());assert len(review['all_dataset_rows'])==15
    evidence=ROOT/'research/clean-global-FRS-prediction-prerequisite-review.json';assert not evidence.exists()
    evidence.write_text(json.dumps({'created_unix':time.time(),'training_review_sha256':digest(REVIEW),
        'all_training_rows':review['all_dataset_rows'],'optimizer_failures':review['optimizer_failures'],
        'training_metrics':review['dataset_balanced_metrics'],'goal_achieved':False,
        'decision':'Run predeclared ALL15 first-inner clean/query/noise screen with full native controls; retain all training failures.'},indent=2),encoding='utf-8')
    base=json.loads((ROOT/'research/clean-global-frs-training-source-freeze.json').read_text())['fingerprints']
    for rel,sha in base.items():
        assert digest(ROOT/rel)==sha,rel
    paths=[p for folder in ('src','scripts','tests') for p in (ROOT/folder).rglob('*.py')]
    paths+=[TESTS,REVIEW,evidence,ROOT/'research/clean-global-FRS-prediction-protocol.md']
    for folder in (SOURCE,NATIVE):
        paths+=[folder/'complete.json',folder/'complete-audit.json']
        paths+=[folder/(name+'.json') for name in sorted(parents[folder.name]['artifact_hashes'])]
    hashes={**base,**{p.relative_to(ROOT).as_posix():digest(p) for p in paths}}
    FREEZE.write_text(json.dumps({'created_unix':time.time(),'fingerprints':hashes,'goal_achieved':False,
        'scope':'ALL15_first_inner_nominal_vs_robust_training_loss_nativefullbudget_query16noise'},indent=2),encoding='utf-8')
    execute('smoke_scoring','run_clean_global_frs_prediction.py',('--dataset','iris'))
    execute('smoke_auditing','audit_clean_global_frs_prediction.py',('--dataset','iris'))
    smoke=json.loads((FOLDER/'iris.smoke-audit.json').read_text());assert smoke['passed'] and smoke['sha256']==digest(FOLDER/'iris.json')
    execute('scoring','run_clean_global_frs_prediction.py')
    execute('auditing','audit_clean_global_frs_prediction.py')
    audit=verify(FOLDER);summary=json.loads((FOLDER/'complete.json').read_text())
    report={'audit_sha256':digest(FOLDER/'complete-audit.json'),'all_dataset_rows':summary['rows'],'means':summary['means'],
        'nominal_clean_BA_differences_pp':summary['nominal_clean_BA_differences_pp'],'nominal_wins_losses':summary['nominal_wins_losses'],
        'mean_actual_feature_counts':summary['mean_actual_feature_counts'],'audit_counts':audit['counts'],
        'classification_scored':True,'classification_superiority_established':False,'new_outer_evaluation_performed':False,
        'goal_achieved':False,'limits':summary['limits'],
        'decision':'Review actual all15 prediction/cost tradeoffs; no automatic promotion or significance claim on exposed data.'}
    (ROOT/'research/clean-global-FRS-prediction-review.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    status('complete_audited_reviewed',audit_sha256=report['audit_sha256'])


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        status('failed',error=str(error));traceback.print_exc();raise
