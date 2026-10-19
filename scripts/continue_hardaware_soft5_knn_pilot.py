"""Monitor live prerequisites; run frozen anisotropic pilot only after audits."""
import argparse
import ctypes
from ctypes import wintypes
import json
import subprocess
import sys
import time

from run_hardaware_soft5_knn_pilot import ROOT, FOLDER, digest
from continue_soft_topk_knn_pilot import verify

STATUS = ROOT/'results/hardaware-soft5-knn-chain-status.json'


def status(phase, **values):
    STATUS.write_text(json.dumps({'phase': phase, 'updated_unix': time.time(),
                                 **values}, indent=2), encoding='utf-8')


def wait_verified(folder, pid):
    ready = folder/'complete-audit.json'
    if ready.exists(): return verify(folder)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetExitCodeProcess.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle: raise RuntimeError(f'No live prerequisite or completed audit: {folder}')
    try:
        while not ready.exists():
            code = wintypes.DWORD()
            assert kernel.GetExitCodeProcess(handle, ctypes.byref(code))
            if code.value != 259:
                raise RuntimeError(f'Prerequisite ended without successful audit: {folder}')
            time.sleep(2)
    finally: kernel.CloseHandle(handle)
    return verify(folder)


def phase(name, script, log):
    with (ROOT/'results'/log).open('w', encoding='utf-8') as stream:
        child = subprocess.Popen([sys.executable, str(ROOT/'scripts'/script)], cwd=ROOT,
                                 stdout=stream, stderr=subprocess.STDOUT)
        status(name, child_pid=child.pid, script=script, log=log)
        if child.wait(): raise RuntimeError(f'{script} failed; see {log}')


def main(training_pid, predictive_pid):
    status('waiting_for_independent_prerequisite_audits', training_pid=training_pid,
           predictive_pid=predictive_pid, goal_achieved=False)
    training = ROOT/'results/hardaware-guarded-soft5-admission'
    prior = ROOT/'results/joint-guarded-soft5-knn-pilot'
    a = wait_verified(training, training_pid)
    assert a['counts']['supports'] == 86 and a['counts']['datasets'] == 15
    wait_verified(prior, predictive_pid)
    prerequisites = {str(p.relative_to(ROOT)): digest(p) for p in
                     (training/'complete-audit.json', prior/'complete-audit.json')}
    status('prerequisites_verified', audit_hashes=prerequisites, goal_achieved=False)
    phase('reviewing_complete_training_diagnostic', 'review_hardaware_training_prerequisite.py',
          'hardaware-training-prerequisite-review.log')
    phase('reviewing_complete_previous_prediction', 'review_joint_guarded_soft5_knn_pilot.py',
          'joint-guarded-soft5-knn-pilot-prerequisite-review.log')
    phase('running_hardaware_predictions', 'run_hardaware_soft5_knn_pilot.py',
          'hardaware-soft5-knn-pilot-run.log')
    phase('auditing_hardaware_predictions', 'audit_hardaware_soft5_knn_pilot.py',
          'hardaware-soft5-knn-pilot-audit.log')
    verify(FOLDER)
    phase('reviewing_complete_hardaware_prediction', 'review_hardaware_soft5_knn_pilot.py',
          'hardaware-soft5-knn-pilot-review.log')
    result = json.loads((FOLDER/'complete.json').read_text())
    status('pilot_audit_and_review_complete', pilot_gate_passed=result['pilot_gate_passed'],
           goal_achieved=False, prerequisite_audit_hashes=prerequisites,
           audit_sha256=digest(FOLDER/'complete-audit.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-chain-pid', type=int, required=True)
    parser.add_argument('--predictive-chain-pid', type=int, required=True)
    args = parser.parse_args()
    try: main(args.training_chain_pid, args.predictive_chain_pid)
    except Exception as error:
        status('failed', error=str(error), goal_achieved=False)
        raise
