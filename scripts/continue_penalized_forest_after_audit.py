"""Preserve the preceding audit, then execute/audit the adjusted forest pilot."""
import argparse
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import time
import continue_projected_label_loss_after_audit as driver

ROOT = Path(__file__).resolve().parents[1]
driver.STATUS = ROOT/'results/penalized-forest-chain-status.json'
BASE = ROOT/'results/crossfit-label-loss-recent-pilot'


def verified_previous():
    path = BASE/'complete-audit.json'
    report = json.loads(path.read_text())
    assert report['passed'] and not report['goal_achieved']
    assert report['summary_sha256'] == driver.digest(BASE/'complete.json')
    assert report['auditor_sha256'] == driver.digest(ROOT/'scripts/audit_crossfit_label_loss_recent_pilot.py')
    assert len(report['artifact_hashes']) == report['counts']['datasets'] == 15
    for name, expected in report['artifact_hashes'].items():
        path = BASE/(name+'.json')
        assert driver.digest(path) == expected
        saved = json.loads(path.read_text())
        for rel, sha in saved['fingerprints'].items():
            assert driver.digest(ROOT/rel) == sha, rel
    return driver.digest(BASE/'complete-audit.json')


def wait_previous(pid):
    if (BASE/'complete-audit.json').exists():
        return verified_previous()
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetExitCodeProcess.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        raise RuntimeError('previous auditor missing without completed evidence')
    driver.write_status('waiting_for_live_prerequisite', prerequisite_pid=pid)
    try:
        while True:
            if (BASE/'complete-audit.json').exists():
                try:
                    return verified_previous()
                except json.JSONDecodeError:
                    pass
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                raise RuntimeError('cannot inspect prerequisite process')
            if code.value != 259:
                if (BASE/'complete-audit.json').exists():
                    return verified_previous()
                raise RuntimeError(f'previous audit exit{code.value} without verified evidence')
            time.sleep(5.)
    finally:
        kernel.CloseHandle(handle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prerequisite-pid', required=True, type=int)
    args = parser.parse_args()
    try:
        sha = wait_previous(args.prerequisite_pid)
        driver.verified_prerequisite()  # Original baseline cache's independent audit.
        driver.run_phase('running_new_pilot', 'run_penalized_forest_recent_pilot.py',
                         'penalized-forest-recent-pilot-run.log', sha)
        driver.run_phase('auditing_new_pilot', 'audit_penalized_forest_recent_pilot.py',
                         'penalized-forest-recent-pilot-audit-run.log', sha)
        folder = ROOT/'results/penalized-forest-recent-pilot'
        audit = json.loads((folder/'complete-audit.json').read_text())
        assert audit['passed'] and not audit['goal_achieved']
        assert audit['summary_sha256'] == driver.digest(folder/'complete.json')
        driver.write_status('pilot_and_audit_complete', prerequisite_sha256=sha,
                            pilot_gate_passed=audit['pilot_gate_passed'],
                            audit_sha256=driver.digest(folder/'complete-audit.json'))
    except Exception as error:
        driver.write_status('failed', error=str(error))
        raise
