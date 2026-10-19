"""Require audited complete reviews, then smoke/ALL15 audit fair28choices."""
import json
import subprocess
import sys
import time
import traceback
from run_monotone_frs_inner import ROOT, FOLDER, PREVIOUS, frozen, digest
from continue_soft_topk_knn_pilot import verify

STATUS = ROOT/'results/monotone-frs-inner-chain-status.json'


def status(phase, **values):
    STATUS.write_text(json.dumps({'phase':phase,'updated_unix':time.time(),'goal_achieved':False,**values},indent=2),encoding='utf-8')


def execute(phase, script, args=()):
    frozen(); log=ROOT/'results'/f'monotone-frs-inner-{phase}.log'
    with log.open('a',encoding='utf-8') as stream:
        child=subprocess.Popen([sys.executable,str(ROOT/'scripts'/script),*args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        status(phase,child_pid=child.pid,log=log.relative_to(ROOT).as_posix(),script_sha256=digest(ROOT/'scripts'/script))
        if child.wait():
            raise RuntimeError(f'{script} failed; retain logs and artifacts')


def main():
    frozen()
    version_hashes=json.loads((ROOT/'research/monotone-frs-inner-review-v2-provenance.json').read_text())['fingerprints']
    for rel,sha in version_hashes.items():
        assert digest(ROOT/rel)==sha,rel
    required=[(PREVIOUS,ROOT/'results/frs-cell-inner-v3-chain-status.json',ROOT/'research/frs-cell-inner-review.json'),
        (ROOT/'results/frs-one-sided-training',ROOT/'results/frs-one-sided-training-v2-chain-status.json',ROOT/'research/frs-one-sided-training-review.json')]
    status('waiting_for_two_complete_independent_reviews')
    while not all((folder/'complete-audit.json').exists() and review.exists() for folder,st,review in required):
        for folder,st,review in required:
            prior_status=json.loads(st.read_text())
            if prior_status['phase']=='failed':
                raise RuntimeError(f'Prerequisite failed: {folder}; no28choicecohort started')
        time.sleep(10)
    evidence={}
    for folder,st,review in required:
        audit=verify(folder);r=json.loads(review.read_text())
        assert r['audit_sha256']==digest(folder/'complete-audit.json')
        evidence[folder.relative_to(ROOT).as_posix()]={'audit_sha256':digest(folder/'complete-audit.json'),'review_sha256':digest(review)}
    execute('alignment_review','review_one_sided_alignment.py')
    alignment=json.loads((ROOT/'research/one-sided-FRS-alignment-review.json').read_text())
    assert alignment['counts']['optimizer_failures']==0 and alignment['counts']['target_expansion_folds']>0
    status('prerequisites_reviewed_no_outer_promotion',evidence=evidence)
    execute('smoke_running','run_monotone_frs_inner.py',('--dataset','iris'))
    execute('smoke_auditing','audit_monotone_frs_inner.py',('--dataset','iris'))
    smoke=json.loads((FOLDER/'iris.smoke-audit.json').read_text())
    assert smoke['passed'] and smoke['sha256']==digest(FOLDER/'iris.json')
    execute('running','run_monotone_frs_inner.py')
    execute('auditing','audit_monotone_frs_inner.py')
    execute('reviewing','review_monotone_frs_inner_v2.py')
    audit=verify(FOLDER);review=json.loads((ROOT/'research/monotone-frs-inner-review.json').read_text())
    assert review['audit_sha256']==digest(FOLDER/'complete-audit.json')
    status('complete_audited_reviewed',pilot_gate_passed=review['pilot_gate_passed'],audit_sha256=review['audit_sha256'])


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        status('failed',error=str(error));traceback.print_exc();raise
