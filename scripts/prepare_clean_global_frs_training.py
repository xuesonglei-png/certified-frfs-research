"""Wait full source review, freeze before fits, then ALL15 and independent audit."""
import json
import subprocess
import sys
import time
import traceback
from run_clean_global_frs_training import ROOT,SOURCE,FOLDER,FREEZE,REVIEW,prerequisites,frozen,digest
from continue_soft_topk_knn_pilot import verify

STATUS=ROOT/'results/clean-global-frs-training-chain-status.json'
TESTS=ROOT/'research/clean-global-FRS-training-tests.json'
PROTOTYPE=ROOT/'research/clean-global-transition-prototype-tests.json'


def status(phase,**values):
    STATUS.write_text(json.dumps({'phase':phase,'updated_unix':time.time(),'goal_achieved':False,
        'classification_scored':False,**values},indent=2),encoding='utf-8')


def execute(phase,script,args=()):
    frozen();log=ROOT/'results'/f'clean-global-frs-training-{phase}.log'
    with log.open('a',encoding='utf-8') as stream:
        child=subprocess.Popen([sys.executable,str(ROOT/'scripts'/script),*args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        status(phase,child_pid=child.pid,script_sha256=digest(ROOT/'scripts'/script))
        if child.wait():
            raise RuntimeError(f'{script} failed; retain every artifact and require a new version for fixes')


def main():
    assert not FREEZE.exists() and not FOLDER.exists()
    for path in (TESTS,PROTOTYPE):
        packet=json.loads(path.read_text());assert packet['passed'] and packet['tests_passed']==2
        for rel,sha in packet['fingerprints'].items():
            assert digest(ROOT/rel)==sha,rel
    status('waiting_for_full_sparse_global_training_review')
    while not REVIEW.exists():
        for name in ('sparse-global-frs-training-chain-status.json','sparse-global-frs-training-preparation-status.json'):
            path=ROOT/'results'/name
            if path.exists() and json.loads(path.read_text())['phase']=='failed':
                raise RuntimeError(f'{name} failed; no new fitting started')
        time.sleep(10)
    parent=prerequisites();review=json.loads(REVIEW.read_text());assert len(review['all_dataset_rows'])==15
    evidence=ROOT/'research/clean-global-FRS-prerequisite-review.json'
    assert not evidence.exists()
    evidence.write_text(json.dumps({'created_unix':time.time(),'review_sha256':digest(REVIEW),
        'all_prior_dataset_rows':review['all_dataset_rows'],'prior_dataset_balanced_metrics':review['dataset_balanced_metrics'],
        'prior_failures':review['optimizer_failures'],'prior_available_regions':review['available_regions'],
        'goal_achieved':False,'classification_scored':False,
        'decision':'Run predeclared matched nominal-versus-robust loss experiment; keep all source outcomes and original constraints.'},indent=2),encoding='utf-8')
    base=json.loads((ROOT/'research/sparse-global-frs-training-source-freeze.json').read_text())['fingerprints']
    for rel,sha in base.items():
        assert digest(ROOT/rel)==sha,rel
    paths=[p for folder in ('src','scripts','tests') for p in (ROOT/folder).rglob('*.py')]
    paths+=[TESTS,PROTOTYPE,evidence,REVIEW,ROOT/'research/clean-global-FRS-training-protocol.md',SOURCE/'complete.json',SOURCE/'complete-audit.json']
    paths+=[SOURCE/(name+'.json') for name in sorted(parent['artifact_hashes'])]
    hashes={**base,**{p.relative_to(ROOT).as_posix():digest(p) for p in paths}}
    FREEZE.write_text(json.dumps({'created_unix':time.time(),'fingerprints':hashes,'goal_achieved':False,
        'scope':'ALL15firsttrainingfoldmatched3startsnominalobjectiveunchangedrobustFRSconstraints'},indent=2),encoding='utf-8')
    execute('smoke_training','run_clean_global_frs_training.py',('--dataset','iris'))
    execute('smoke_auditing','audit_clean_global_frs_training.py',('--dataset','iris'))
    smoke=json.loads((FOLDER/'iris.smoke-audit.json').read_text());assert smoke['passed'] and smoke['sha256']==digest(FOLDER/'iris.json')
    execute('training','run_clean_global_frs_training.py')
    execute('auditing','audit_clean_global_frs_training.py')
    audit=verify(FOLDER);summary=json.loads((FOLDER/'complete.json').read_text())
    final={'audit_sha256':digest(FOLDER/'complete-audit.json'),'all_dataset_rows':summary['rows'],
        'dataset_balanced_metrics':summary['dataset_balanced_metrics'],'optimizer_failures':summary['optimizer_failures'],
        'available_global_regions':summary['available_global_regions'],'folds_with_exact_zero_weights':summary['folds_with_exact_zero_weights'],
        'audit_counts':audit['counts'],'goal_achieved':False,'classification_scored':False,'limits':summary['limits'],
        'decision':'Assess both training losses and sparsity; actual native-controlled prediction still required.'}
    (ROOT/'research/clean-global-FRS-training-review.json').write_text(json.dumps(final,indent=2),encoding='utf-8')
    status('complete_audited_reviewed',audit_sha256=final['audit_sha256'])


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        status('failed',error=str(error));traceback.print_exc();raise
