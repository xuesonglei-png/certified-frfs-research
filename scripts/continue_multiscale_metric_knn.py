"""Wait for all15admissions, audit, then run and audit the predictive pilot."""
import argparse
import ctypes
from ctypes import wintypes
import json
import subprocess
import sys
import time
from pathlib import Path
from run_recent_multiscale_pool_admission import ROOT, FOLDER as ADMISSION, digest

STATUS = ROOT/'results/multiscale-metric-knn-chain-status.json'


def status(phase, **values):
    STATUS.write_text(json.dumps({'phase': phase, 'updated_unix': time.time(), **values}, indent=2), encoding='utf-8')


def wait_admission(pid):
    summary = ADMISSION/'complete.json'
    if summary.exists(): return
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetExitCodeProcess.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle: raise RuntimeError('Admission absent and no live prerequisite process.')
    status('waiting_for_admission', prerequisite_pid=pid)
    try:
        while not summary.exists():
            code = wintypes.DWORD(); assert kernel.GetExitCodeProcess(handle, ctypes.byref(code))
            if code.value != 259: raise RuntimeError('Admission exited without complete all15evidence.')
            time.sleep(2.)
    finally: kernel.CloseHandle(handle)


def phase(name, script, log):
    with (ROOT/'results'/log).open('w', encoding='utf-8') as stream:
        child = subprocess.Popen([sys.executable, str(ROOT/'scripts'/script)], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        status(name, child_pid=child.pid, script=script, log=log)
        code = child.wait()
        if code: raise RuntimeError(f'{script} exited {code}; see {log}')


def main(pid):
    wait_admission(pid)
    summary = json.loads((ADMISSION/'complete.json').read_text())
    assert len(summary['rows']) == 15 and not summary['goal_achieved']
    phase('auditing_admission', 'audit_recent_multiscale_pool_admission.py', 'recent-multiscale-pool-admission-audit.log')
    audit = json.loads((ADMISSION/'complete-audit.json').read_text())
    assert audit['passed'] and audit['summary_sha256'] == digest(ADMISSION/'complete.json')
    phase('running_multiscale_predictions', 'run_multiscale_certified_metric_knn_pool.py', 'multiscale-certified-metric-knn-pool-run.log')
    phase('auditing_multiscale_predictions', 'audit_multiscale_certified_metric_knn_pool.py', 'multiscale-certified-metric-knn-pool-audit.log')
    folder = ROOT/'results/multiscale-certified-metric-knn-pool'
    audit = json.loads((folder/'complete-audit.json').read_text()); result = json.loads((folder/'complete.json').read_text())
    assert audit['passed'] and audit['summary_sha256'] == digest(folder/'complete.json')
    status('pilot_and_audit_complete', pilot_gate_passed=result['pilot_gate_passed'], goal_achieved=False,
           audit_sha256=digest(folder/'complete-audit.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prerequisite-pid', required=True, type=int); args = parser.parse_args()
    try: main(args.prerequisite_pid)
    except Exception as error:
        status('failed', error=str(error)); raise
