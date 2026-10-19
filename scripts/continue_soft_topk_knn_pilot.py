"""Wait for audited predecessor, then run/audit the equal-budget soft5 pilot."""
import argparse
import ctypes
from ctypes import wintypes
import json
import subprocess
import sys
import time

from run_soft_topk_knn_pilot import ROOT, FOLDER, PARENT, digest

STATUS = ROOT/'results/soft-topk-knn-chain-status.json'


def status(phase, **values):
    STATUS.write_text(json.dumps({'phase': phase, 'updated_unix': time.time(), **values}, indent=2), encoding='utf-8')


def verify(folder):
    audit = json.loads((folder/'complete-audit.json').read_text())
    assert audit['passed'] and audit['summary_sha256'] == digest(folder/'complete.json')
    for name, sha in audit['artifact_hashes'].items(): assert digest(folder/(name+'.json')) == sha
    return audit


def wait_for_previous(pid):
    previous = ROOT/'results/majority-predictive-knn-pilot'; ready = previous/'complete-audit.json'
    if ready.exists(): return verify(previous)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]; kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]; kernel.GetExitCodeProcess.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle: raise RuntimeError('No prior audit evidence or live audit process')
    try:
        while not ready.exists():
            code = wintypes.DWORD(); assert kernel.GetExitCodeProcess(handle, ctypes.byref(code))
            if code.value != 259: raise RuntimeError('Prior audit ended without completion evidence')
            time.sleep(2.)
    finally:
        kernel.CloseHandle(handle)
    return verify(previous)


def phase(name, script, log):
    with (ROOT/'results'/log).open('w', encoding='utf-8') as stream:
        child = subprocess.Popen([sys.executable, str(ROOT/'scripts'/script)], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        status(name, child_pid=child.pid, script=script, log=log)
        if child.wait(): raise RuntimeError(f'{script} failed; see {log}')


def main(pid):
    status('waiting_for_predecessor_audit', prerequisite_pid=pid)
    wait_for_previous(pid); verify(PARENT)
    training = verify(ROOT/'results/soft-topk-target-admission')
    assert training['counts']['supports'] == 86 and training['counts']['optimizer_failures'] == 0
    phase('running_soft5_predictions', 'run_soft_topk_knn_pilot.py', 'soft-topk-knn-pilot-run.log')
    phase('auditing_soft5_predictions', 'audit_soft_topk_knn_pilot.py', 'soft-topk-knn-pilot-audit.log')
    verify(FOLDER); result = json.loads((FOLDER/'complete.json').read_text())
    status('pilot_and_audit_complete', pilot_gate_passed=result['pilot_gate_passed'], goal_achieved=False,
           audit_sha256=digest(FOLDER/'complete-audit.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--previous-audit-pid', type=int, required=True)
    args = parser.parse_args()
    try: main(args.previous_audit_pid)
    except Exception as error: status('failed', error=str(error)); raise
