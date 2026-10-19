"""Validate both preceding audits, then run and audit new nested predictions."""
import json
from pathlib import Path
import continue_projected_label_loss_after_audit as driver
from continue_penalized_forest_after_audit import verified_previous

ROOT = Path(__file__).resolve().parents[1]
driver.STATUS = ROOT/'results/repaired-crossfit-chain-status.json'


if __name__ == '__main__':
    try:
        sha = verified_previous()
        driver.verified_prerequisite()
        folder = ROOT/'results/flexible-majority-repair-admission'
        admission = json.loads((folder/'complete.json').read_text())
        audit = json.loads((folder/'complete-audit.json').read_text())
        assert audit['passed'] and admission['admission_gate_passed']
        assert audit['summary_sha256'] == driver.digest(folder/'complete.json')
        assert audit['auditor_sha256'] == driver.digest(ROOT/'scripts/audit_flexible_majority_repair_admission.py')
        for row in admission['rows']:
            for rel, fingerprint in row['fingerprints'].items():
                assert driver.digest(ROOT/rel) == fingerprint, rel
        driver.run_phase('running_new_pilot', 'run_repaired_crossfit_recent_pilot.py',
                         'repaired-crossfit-recent-pilot-run.log', sha)
        driver.run_phase('auditing_new_pilot', 'audit_repaired_crossfit_recent_pilot.py',
                         'repaired-crossfit-recent-pilot-audit-run.log', sha)
        folder = ROOT/'results/repaired-crossfit-recent-pilot'
        audit = json.loads((folder/'complete-audit.json').read_text())
        assert audit['passed'] and not audit['goal_achieved']
        assert audit['summary_sha256'] == driver.digest(folder/'complete.json')
        driver.write_status('pilot_and_audit_complete', prerequisite_sha256=sha,
                            pilot_gate_passed=audit['pilot_gate_passed'],
                            audit_sha256=driver.digest(folder/'complete-audit.json'))
    except Exception as error:
        driver.write_status('failed', error=str(error))
        raise
