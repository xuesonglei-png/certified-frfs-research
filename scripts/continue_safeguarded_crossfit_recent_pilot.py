"""Wait on v2 audit evidence, then run and audit guarded all15 predictions."""
import argparse
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import time
import numpy as np
import continue_projected_label_loss_after_audit as driver

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'results/repaired-crossfit-recent-pilot'
driver.STATUS = ROOT/'results/safeguarded-crossfit-chain-status.json'


def prerequisite():
    path = BASE/'complete-audit-v2.json'
    report = json.loads(path.read_text())
    assert report['passed'] and not report['goal_achieved']
    assert report['summary_sha256'] == driver.digest(BASE/'complete.json')
    assert report['auditor_sha256'] == driver.digest(ROOT/'scripts/audit_repaired_crossfit_recent_pilot_v2.py')
    for name, sha in report['artifact_hashes'].items():
        assert driver.digest(BASE/(name+'.json')) == sha
    return driver.digest(path)


def wait_previous(pid):
    path = BASE/'complete-audit-v2.json'
    if path.exists():
        return prerequisite()
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetExitCodeProcess.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        raise RuntimeError('previous audit missing without completed evidence')
    driver.write_status('waiting_for_live_prerequisite', prerequisite_pid=pid)
    try:
        while True:
            if path.exists():
                try:
                    return prerequisite()
                except json.JSONDecodeError:
                    pass
            code = wintypes.DWORD()
            assert kernel.GetExitCodeProcess(handle, ctypes.byref(code))
            if code.value != 259:
                if path.exists():
                    return prerequisite()
                raise RuntimeError('previous audit exited without verified evidence')
            time.sleep(5.)
    finally:
        kernel.CloseHandle(handle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prerequisite-pid', required=True, type=int)
    args = parser.parse_args()
    try:
        sha = wait_previous(args.prerequisite_pid)
        driver.verified_prerequisite()
        folder = ROOT/'results/classwise-repair-admission'
        saved = json.loads((folder/'complete.json').read_text())
        audit = json.loads((folder/'complete-audit.json').read_text())
        assert audit['passed'] and saved['admission_gate_passed']
        assert audit['summary_sha256'] == driver.digest(folder/'complete.json')
        assert audit['auditor_sha256'] == driver.digest(ROOT/'scripts/audit_classwise_repair_admission.py')
        for row in saved['rows']:
            assert json.loads((folder/(row['dataset']+'.json')).read_text()) == row
            for rel, fingerprint in row['fingerprints'].items():
                assert driver.digest(ROOT/rel) == fingerprint, rel
        np.testing.assert_allclose(saved['mean_removal'], np.mean([1-r['actual_count']/r['d'] for r in saved['rows']]), atol=1e-12)
        driver.run_phase('running_new_pilot', 'run_safeguarded_crossfit_recent_pilot.py',
                         'safeguarded-crossfit-recent-pilot-run.log', sha)
        driver.run_phase('auditing_new_pilot', 'audit_safeguarded_crossfit_recent_pilot.py',
                         'safeguarded-crossfit-recent-pilot-audit-run.log', sha)
        folder = ROOT/'results/safeguarded-crossfit-recent-pilot'
        audit = json.loads((folder/'complete-audit.json').read_text())
        assert audit['passed'] and not audit['goal_achieved']
        assert audit['summary_sha256'] == driver.digest(folder/'complete.json')
        driver.write_status('pilot_and_audit_complete', prerequisite_sha256=sha,
                            pilot_gate_passed=audit['pilot_gate_passed'],
                            audit_sha256=driver.digest(folder/'complete-audit.json'))
    except Exception as error:
        driver.write_status('failed', error=str(error))
        raise
