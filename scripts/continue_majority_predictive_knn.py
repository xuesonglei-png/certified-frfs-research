"""Run the dual-target predictive pilot and independent audit in sequence."""
import json
import subprocess
import sys
import time

from run_majority_predictive_knn_pilot import ROOT, FOLDER, PARENT, digest

STATUS = ROOT/'results/majority-predictive-knn-chain-status.json'


def status(phase, **values):
    STATUS.write_text(json.dumps({'phase': phase, 'updated_unix': time.time(), **values}, indent=2), encoding='utf-8')


def phase(name, script, log):
    with (ROOT/'results'/log).open('w', encoding='utf-8') as stream:
        child = subprocess.Popen([sys.executable, str(ROOT/'scripts'/script)], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        status(name, child_pid=child.pid, script=script, log=log)
        if child.wait(): raise RuntimeError(f'{script} failed; see {log}')


def verify(folder):
    audit = json.loads((folder/'complete-audit.json').read_text())
    assert audit['passed'] and audit['summary_sha256'] == digest(folder/'complete.json')
    for name, sha in audit['artifact_hashes'].items(): assert digest(folder/(name+'.json')) == sha
    return audit


def main():
    verify(PARENT)
    training = verify(ROOT/'results/majority-constrained-target-admission-v3')
    assert training['counts']['supports'] == 86 and training['counts']['admitted_nontrivial_mixtures'] == 252
    phase('running_dual_target_predictions', 'run_majority_predictive_knn_pilot.py', 'majority-predictive-knn-pilot-run.log')
    phase('auditing_dual_target_predictions', 'audit_majority_predictive_knn_pilot.py', 'majority-predictive-knn-pilot-audit.log')
    audit = verify(FOLDER); result = json.loads((FOLDER/'complete.json').read_text())
    status('pilot_and_audit_complete', pilot_gate_passed=result['pilot_gate_passed'], goal_achieved=False,
           audit_sha256=digest(FOLDER/'complete-audit.json'))


if __name__ == '__main__':
    try: main()
    except Exception as error: status('failed', error=str(error)); raise
