"""V2 summary-only numpy-int JSON repair; preserve original failure evidence."""
import json
import subprocess
import sys
import time
import traceback
import numpy as np
from diagnose_classwise_star_admission import ROOT, FOLDER, SOURCE, digest
from continue_soft_topk_knn_pilot import verify

STATUS = ROOT/'results/classwise-star-admission-recovery-v2-status.json'


def status(phase, **values):
    STATUS.write_text(json.dumps({'phase': phase, 'updated_unix': time.time(), 'goal_achieved': False, **values}, indent=2), encoding='utf-8')


def main():
    original_status = json.loads((ROOT/'results/classwise-star-admission-chain-status.json').read_text())
    assert original_status['phase'] == 'failed'
    names = sorted(verify(SOURCE)['artifact_hashes']); assert len(names) == 15
    provenance = {'original_failed_status': original_status,
                  'original_failure_log_sha256': digest(ROOT/'results/classwise-star-admission-running.log'),
                  'scope': 'summary_numpy_int64_JSON_encoding_only_no_training_bound_or_prediction_change'}
    rows = []; folds = 0
    for name in names:
        saved = json.loads((FOLDER/(name+'.json')).read_text()); assert saved['complete']
        assert not saved['classification_scored']; folds += len(saved['inner'])
        for rel, sha in saved['fingerprints'].items():
            assert digest(ROOT/rel) == sha, rel
        checks = [r for b in saved['inner'] for r in b['checks']]
        # Explicit builtin int conversion is the complete repair. No bound,
        # feasibility flag or dataset artifact is regenerated or altered.
        rows.append({'dataset': name, 'guard_checks': len(checks),
            'new_admissions': int(sum(r['new_admission'] for r in checks)),
            'class_bounds_strictly_tightened': int(sum(np.sum(np.asarray(r['certificate']['class_certificate']['upper_bounds'])
                < np.asarray(r['certificate']['original_class_certificate']['upper_bounds'])-1e-12) for r in checks)),
            'total_class_bounds': int(sum(len(r['certificate']['class_certificate']['upper_bounds']) for r in checks))})
    summary = {'rows': rows, 'counts': {k: int(sum(r[k] for r in rows)) for k in
        ('guard_checks', 'new_admissions', 'class_bounds_strictly_tightened', 'total_class_bounds')},
        'datasets': len(names), 'inner_folds': folds, 'artifact_hashes': {n: digest(FOLDER/(n+'.json')) for n in names},
        'goal_achieved': False, 'independent_audit_required': True, 'classification_scored': False,
        'scope': 'certificate_tightness_and_admission_not_prediction_superiority', 'recovery_provenance': provenance}
    complete = FOLDER/'complete.json'
    if complete.exists():
        assert json.loads(complete.read_text()) == summary
    else:
        complete.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    log = ROOT/'results/classwise-star-admission-auditing-v2.log'
    with log.open('a', encoding='utf-8') as stream:
        child = subprocess.Popen([sys.executable, str(ROOT/'scripts/audit_classwise_star_admission.py')], cwd=ROOT,
                                 stdout=stream, stderr=subprocess.STDOUT)
        status('auditing_recovered_summary', child_pid=child.pid, log=log.relative_to(ROOT).as_posix())
        if child.wait():
            raise RuntimeError('original independent bound auditor failed; retain unchanged summary and all failures')
    audit = verify(FOLDER)
    review = {'counts': summary['counts'], 'audit_sha256': digest(FOLDER/'complete-audit.json'), 'recovery_provenance': provenance,
        'goal_achieved': False, 'classification_scored': False,
        'scope': 'classwise_FRS_tighter_certificate_not_prediction_advantage',
        'decision': 'Two-stage evaluate newly feasible metrics with identical treatment for all methods.' if summary['counts']['new_admissions']
                    else 'No feasible expansion observed; do not promote symbolic example to classification gains.',
        'limits': ['Finite-table class dependency only, not prediction accuracy.', 'Ordinary float conservatism, not directed rounding.',
                   'Only stable unsaturated cells have exactness; originality unverified.']}
    (ROOT/'research/classwise-star-admission-review-v2.json').write_text(json.dumps(review, indent=2), encoding='utf-8')
    status('complete_audited_reviewed', counts=summary['counts'], original_failure_preserved=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed', error=str(error)); traceback.print_exc(); raise
