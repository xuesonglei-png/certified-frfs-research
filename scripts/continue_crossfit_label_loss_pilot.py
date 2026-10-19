"""Run the frozen OOF pilot and audit after checking the preceding audit."""
import json
from pathlib import Path
from continue_projected_label_loss_after_audit import digest, verified_prerequisite, run_phase
import continue_projected_label_loss_after_audit as driver

ROOT = Path(__file__).resolve().parents[1]
driver.STATUS = ROOT/'results/crossfit-label-loss-chain-status.json'


if __name__ == '__main__':
    try:
        sha = verified_prerequisite()
        previous = ROOT/'results/projected-label-loss-recent-pilot'
        audit = json.loads((previous/'complete-audit.json').read_text())
        assert audit['passed'] and not audit['goal_achieved']
        assert audit['auditor_sha256'] == digest(ROOT/'scripts/audit_projected_label_loss_recent_pilot.py')
        assert audit['summary_sha256'] == digest(previous/'complete.json')
        for name, expected in audit['artifact_hashes'].items():
            assert digest(previous/(name+'.json')) == expected
        driver.write_status('prerequisite_verified', prerequisite_sha256=sha)
        run_phase('running_new_pilot', 'run_crossfit_label_loss_recent_pilot.py',
                  'crossfit-label-loss-recent-pilot-run.log', sha)
        run_phase('auditing_new_pilot', 'audit_crossfit_label_loss_recent_pilot.py',
                  'crossfit-label-loss-recent-pilot-audit-run.log', sha)
        folder = ROOT/'results/crossfit-label-loss-recent-pilot'
        report = json.loads((folder/'complete-audit.json').read_text())
        assert report['passed'] and not report['goal_achieved']
        assert report['summary_sha256'] == digest(folder/'complete.json')
        driver.write_status('pilot_and_audit_complete', prerequisite_sha256=sha,
                            pilot_gate_passed=report['pilot_gate_passed'],
                            audit_sha256=digest(folder/'complete-audit.json'))
    except Exception as error:
        driver.write_status('failed', error=str(error))
        raise
