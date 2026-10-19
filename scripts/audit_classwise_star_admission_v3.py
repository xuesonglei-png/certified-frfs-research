"""Unchanged per-dataset star verification; JSON-native aggregate counts.

Original run and V2 summary recovery both failed at output serialization.
Their files and logs remain unchanged. Every mathematical check runs again.
"""
import json
import time
import traceback
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from audit_classwise_star_admission import audit, ROOT, FOLDER, digest, equivalent

STATUS = ROOT/'results/classwise-star-admission-audit-v3-status.json'


def status(phase, **values):
    STATUS.write_text(json.dumps({'phase': phase, 'updated_unix': time.time(), 'goal_achieved': False, **values}, indent=2), encoding='utf-8')


def main():
    summary = json.loads((FOLDER/'complete.json').read_text()); reports = []; counts = Counter()
    status('independently_replaying_unchanged_star_checks')
    with ProcessPoolExecutor(max_workers=2) as executor:
        for f in as_completed([executor.submit(audit, r['dataset']) for r in summary['rows']]):
            r = f.result(); r['counts'] = {k: int(v) for k, v in r['counts'].items()}
            (FOLDER/(r['dataset']+'.independent-v3.json')).write_text(json.dumps(r, indent=2), encoding='utf-8')
            reports.append(r); counts.update(r['counts'])
    assert counts['datasets'] == 15 and counts['inner_folds'] == 44
    for k, v in summary['counts'].items():
        assert counts[k] == v, k
    hashes = {r['dataset']: r['sha256'] for r in reports}; equivalent(hashes, summary['artifact_hashes'])
    provenance = {'original_run_failed_status': json.loads((ROOT/'results/classwise-star-admission-chain-status.json').read_text()),
                  'summary_recovery_failed_status': json.loads((ROOT/'results/classwise-star-admission-recovery-v2-status.json').read_text()),
                  'original_audit_log_sha256': digest(ROOT/'results/classwise-star-admission-auditing-v2.log'),
                  'reused_numerical_auditor_sha256': digest(ROOT/'scripts/audit_classwise_star_admission.py'),
                  'repair': 'Aggregate count conversion to builtin int only; all mathematical verifications unchanged and rerun.'}
    result = {'passed': True, 'counts': dict(counts), 'artifact_hashes': hashes, 'summary_sha256': digest(FOLDER/'complete.json'),
        'auditor_sha256': digest(Path(__file__)), 'recovery_provenance': provenance, 'goal_achieved': False, 'classification_scored': False,
        'scope': 'independent_scalar_knot_stability_and_class_mean_star_bounds_not_prediction_or_population_risk'}
    path = FOLDER/'complete-audit.json'; assert not path.exists()
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    review = {'counts': summary['counts'], 'audit_sha256': digest(path), 'recovery_provenance': provenance,
        'goal_achieved': False, 'classification_scored': False,
        'scope': 'classwise_FRS_tighter_certificate_not_prediction_advantage',
        'decision': 'Only two newly admitted metrics. Quantify extra utility before classifier promotion.'}
    (ROOT/'research/classwise-star-admission-review-v3.json').write_text(json.dumps(review, indent=2), encoding='utf-8')
    status('complete_audited_reviewed', counts=summary['counts']); print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed', error=str(error)); traceback.print_exc(); raise
