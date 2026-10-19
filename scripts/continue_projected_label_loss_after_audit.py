"""Run the next experiment only after a verified live prerequisite succeeds.

This local batch driver waits on an open handle to the existing audit process,
not a lock file, and validates the finished evidence before launching either
the new pilot or its independent auditor. It does not restart an old job.
"""
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT/'.venv/Scripts/python.exe'
BASE = ROOT/'results/adaptive-multiknot-recent-pilot'
STATUS = ROOT/'results/projected-label-loss-chain-status.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_status(phase, **fields):
    data = {'phase': phase, 'goal_achieved': False, **fields}
    temporary = STATUS.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
    os.replace(temporary, STATUS)
    print(data, flush=True)


def verified_prerequisite():
    path = BASE/'complete-audit.json'
    audit = json.loads(path.read_text())
    assert audit['passed'] and not audit['goal_achieved']
    assert audit['auditor_sha256'] == digest(ROOT/'scripts/audit_adaptive_multiknot_recent_pilot.py')
    assert audit['summary_sha256'] == digest(BASE/'complete.json')
    assert len(audit['artifact_hashes']) == audit['counts']['datasets'] == 15
    for name, sha in audit['artifact_hashes'].items():
        artifact = BASE/(name+'.json')
        assert digest(artifact) == sha
        per = json.loads((BASE/(name+'-audit.json')).read_text())
        assert per['fingerprints']['artifact_sha256'] == sha
        assert per['fingerprints']['auditor_sha256'] == audit['auditor_sha256']
        saved = json.loads(artifact.read_text())
        assert saved['complete'] and not saved['query_labels_used_for_selection']
        for rel, fingerprint in saved['fingerprints'].items():
            assert digest(ROOT/rel) == fingerprint, rel
    return digest(path)


def wait_on_audit(pid):
    if (BASE/'complete-audit.json').exists():
        try:
            return verified_prerequisite()
        except json.JSONDecodeError:
            pass  # The live coordinator may still be writing its one artifact.
    assert os.name == 'nt', 'this driver uses a Windows process handle'
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetExitCodeProcess.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        raise RuntimeError('prerequisite handle unavailable without completed evidence')
    write_status('waiting_for_live_prerequisite', prerequisite_pid=pid)
    try:
        while True:
            if (BASE/'complete-audit.json').exists():
                try:
                    return verified_prerequisite()
                except json.JSONDecodeError:
                    pass
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                raise RuntimeError('cannot inspect prerequisite process')
            if code.value != 259:
                # Re-read after exit in case the last write completed just now.
                if (BASE/'complete-audit.json').exists():
                    return verified_prerequisite()
                raise RuntimeError(f'prerequisite terminated({code.value}) without verified complete audit')
            time.sleep(5.)
    finally:
        kernel.CloseHandle(handle)


def run_phase(phase, script, log, prerequisite_sha):
    write_status(phase, script=script, prerequisite_sha256=prerequisite_sha)
    with (ROOT/'results'/log).open('w', encoding='utf-8') as output:
        child = subprocess.Popen([str(PYTHON), str(ROOT/'scripts'/script)], cwd=ROOT,
                                 stdout=output, stderr=subprocess.STDOUT)
        write_status(phase, script=script, child_pid=child.pid, prerequisite_sha256=prerequisite_sha)
        code = child.wait()
    if code:
        raise RuntimeError(f'{script} terminated with code{code}; see {log}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prerequisite-pid', type=int, required=True)
    args = parser.parse_args()
    try:
        sha = wait_on_audit(args.prerequisite_pid)
        write_status('prerequisite_verified', prerequisite_sha256=sha)
        run_phase('running_new_pilot', 'run_projected_label_loss_recent_pilot.py',
                  'projected-label-loss-recent-pilot-run.log', sha)
        run_phase('auditing_new_pilot', 'audit_projected_label_loss_recent_pilot.py',
                  'projected-label-loss-recent-pilot-audit-run.log', sha)
        folder = ROOT/'results/projected-label-loss-recent-pilot'
        report = json.loads((folder/'complete-audit.json').read_text())
        assert report['passed'] and not report['goal_achieved']
        assert report['summary_sha256'] == digest(folder/'complete.json')
        write_status('pilot_and_audit_complete', prerequisite_sha256=sha,
                     pilot_gate_passed=report['pilot_gate_passed'], audit_sha256=digest(folder/'complete-audit.json'))
    except Exception as error:
        write_status('failed', error=str(error))
        raise
