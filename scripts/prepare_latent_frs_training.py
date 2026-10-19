"""Queue after FULL capped review; finalize unchanged draft, freeze, fit/audit."""
import json
import subprocess
import sys
import time
import traceback
from run_latent_frs_training import ROOT, SOURCE, REVIEW, FREEZE, prerequisite, digest

STATUS = ROOT/'results/latent-frs-training-preparation-status.json'
DRAFT = ROOT/'research/latent-FRS-training-protocol-draft.md'
PROTOCOL = ROOT/'research/latent-FRS-training-protocol.md'
EVIDENCE = ROOT/'research/latent-FRS-prerequisite-review.json'


def status(phase, **values):
    STATUS.write_text(json.dumps({'phase':phase, 'updated_unix':time.time(), 'goal_achieved':False,
        'classification_scored':False, **values}, indent=2), encoding='utf-8')


def main():
    assert not FREEZE.exists() and not PROTOCOL.exists() and not EVIDENCE.exists()
    draft_sha = digest(DRAFT); status('waiting_for_complete_capped_audit_and_review', draft_sha256=draft_sha)
    while not (SOURCE/'complete-audit.json').exists() or not REVIEW.exists():
        state = json.loads((ROOT/'results/capped-frs-training-chain-status.json').read_text())
        if state['phase'] == 'failed':
            raise RuntimeError('Capped prerequisite failed; no latent fitting started')
        time.sleep(10)
    audit = prerequisite(); review = json.loads(REVIEW.read_text())
    summary = json.loads((SOURCE/'complete.json').read_text())
    assert summary['datasets'] == 15 and summary['inner_training_folds'] == 44
    assert len(review['all_dataset_rows']) == 15 and len(audit['artifact_hashes']) == 15
    assert digest(DRAFT) == draft_sha
    evidence = {'created_unix':time.time(), 'capped_audit_sha256':digest(SOURCE/'complete-audit.json'),
        'capped_review_sha256':digest(REVIEW), 'all_capped_dataset_rows':review['all_dataset_rows'],
        'capped_training_gain_pp':review['mean_dataset_clipped_training_bound_decrease_pp'],
        'capped_optimizer_failures':review['optimizer_failures'],
        'worse_bound_folds':review['worse_bound_folds'], 'better_bound_folds':review['better_bound_folds'],
        'draft_sha256':draft_sha, 'goal_achieved':False, 'classification_scored':False,
        'decision':'Fix unchanged matched-start training-only protocol; no classifier or significance claim.'}
    EVIDENCE.write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    PROTOCOL.write_text(DRAFT.read_text()+ '\nPrerequisite all-row review SHA256: '+digest(EVIDENCE)+'\n', encoding='utf-8')
    for phase, script in (('freezing', 'freeze_latent_frs_training.py'),
                          ('running_frozen_chain', 'continue_latent_frs_training.py')):
        log = ROOT/'results'/f'latent-frs-training-preparation-{phase}.log'
        with log.open('a', encoding='utf-8') as stream:
            child = subprocess.Popen([sys.executable, str(ROOT/'scripts'/script)], cwd=ROOT,
                stdout=stream, stderr=subprocess.STDOUT)
            status(phase, child_pid=child.pid, script_sha256=digest(ROOT/'scripts'/script))
            if child.wait():
                raise RuntimeError(f'{script} failed; retain evidence and log')
    status('complete_audited_reviewed')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed', error=str(error)); traceback.print_exc(); raise
