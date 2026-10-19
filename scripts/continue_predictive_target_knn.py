"""Require audited multiscale baseline, then run and audit supervised targets."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import time
import continue_multiscale_metric_knn as prerequisite_driver
from run_predictive_target_knn_pilot import ROOT, PARENT, digest

STATUS = ROOT/'results/predictive-target-knn-chain-status.json'


def status(phase, **values):
    STATUS.write_text(json.dumps({'phase': phase, 'updated_unix': time.time(), **values}, indent=2), encoding='utf-8')


def phase(name, script, log):
    with (ROOT/'results'/log).open('w', encoding='utf-8') as stream:
        child = subprocess.Popen([sys.executable, str(ROOT/'scripts'/script)], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        status(name, child_pid=child.pid, script=script, log=log)
        code = child.wait()
        if code: raise RuntimeError(f'{script} exited{code};see{log}')


def main(pid):
    prerequisite_driver.STATUS = STATUS
    # Reuse the verified Windows process-handle wait with audit evidence as
    # its completion file. Process remains alive while this audit executes.
    prerequisite_driver.ADMISSION = PARENT
    status('waiting_for_multiscale_audit', prerequisite_pid=pid)
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]; kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]; kernel.GetExitCodeProcess.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    path = PARENT/'complete-audit.json'; handle = None
    if not path.exists():
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle: raise RuntimeError('Previous audit absent and no live prerequisite process.')
    try:
        while not path.exists():
            code = wintypes.DWORD(); assert kernel.GetExitCodeProcess(handle, ctypes.byref(code))
            if code.value != 259: raise RuntimeError('Previous audit exited without completion evidence.')
            time.sleep(2.)
    finally:
        if handle: kernel.CloseHandle(handle)
    audit = json.loads(path.read_text()); assert audit['passed'] and audit['summary_sha256'] == digest(PARENT/'complete.json')
    for name, sha in audit['artifact_hashes'].items(): assert digest(PARENT/(name+'.json')) == sha
    target = ROOT/'results/predictive-target-metric-admission'; audit = json.loads((target/'complete-audit.json').read_text())
    assert audit['passed'] and audit['counts']['admitted_nontrivial_mixtures'] > 0
    assert audit['summary_sha256'] == digest(target/'complete.json')
    phase('running_predictive_target_pilot', 'run_predictive_target_knn_pilot.py', 'predictive-target-knn-pilot-run.log')
    phase('auditing_predictive_target_pilot', 'audit_predictive_target_knn_pilot.py', 'predictive-target-knn-pilot-audit.log')
    folder = ROOT/'results/predictive-target-knn-pilot'; result = json.loads((folder/'complete.json').read_text()); audit = json.loads((folder/'complete-audit.json').read_text())
    assert audit['passed'] and audit['summary_sha256'] == digest(folder/'complete.json')
    status('pilot_and_audit_complete', pilot_gate_passed=result['pilot_gate_passed'], goal_achieved=False, audit_sha256=digest(folder/'complete-audit.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prerequisite-pid', required=True, type=int); args = parser.parse_args()
    try: main(args.prerequisite_pid)
    except Exception as error: status('failed', error=str(error)); raise
