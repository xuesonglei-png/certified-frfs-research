"""Wait on the live diagnostic, audit it once complete, preserve status."""
import argparse
import ctypes
from ctypes import wintypes
import json
import subprocess
import sys
import time

from run_majority_constrained_target_admission import ROOT, FOLDER, digest

STATUS = ROOT/'results/majority-constrained-target-admission-chain-status.json'


def status(phase, **values):
    STATUS.write_text(json.dumps({'phase': phase, 'updated_unix': time.time(), **values}, indent=2), encoding='utf-8')


def main(pid):
    complete = FOLDER/'complete.json'
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]; kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]; kernel.GetExitCodeProcess.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = None
    status('waiting_for_constrained_admission', prerequisite_pid=pid)
    if not complete.exists():
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle: raise RuntimeError('No completion evidence and prerequisite is not live')
    try:
        while not complete.exists():
            code = wintypes.DWORD(); assert kernel.GetExitCodeProcess(handle, ctypes.byref(code))
            if code.value != 259: raise RuntimeError('Prerequisite exited without completion evidence')
            time.sleep(2.)
    finally:
        if handle: kernel.CloseHandle(handle)
    summary = json.loads(complete.read_text())
    assert len(summary['rows']) == 15 and summary['counts']['supports'] == 86
    with (ROOT/'results/majority-constrained-target-admission-v3-audit.log').open('w', encoding='utf-8') as log:
        child = subprocess.Popen([sys.executable, str(ROOT/'scripts/audit_majority_constrained_target_admission.py')],
                                 cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        status('auditing_constrained_admission', child_pid=child.pid)
        if child.wait(): raise RuntimeError('Independent constrained diagnostic audit failed; see audit log')
    audit = json.loads((FOLDER/'complete-audit.json').read_text())
    assert audit['passed'] and audit['summary_sha256'] == digest(complete)
    status('admission_and_audit_complete', counts=summary['counts'], goal_achieved=False,
           audit_sha256=digest(FOLDER/'complete-audit.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prerequisite-pid', type=int, required=True)
    args = parser.parse_args()
    try: main(args.prerequisite_pid)
    except Exception as error: status('failed', error=str(error)); raise
