"""Run, independently audit and review training-only hard-aware cohort."""
import json
import subprocess
import sys
import time
from run_hardaware_guarded_soft5_admission import ROOT


def main():
    status = ROOT/'results/hardaware-training-chain-status.json'
    for phase, command in [
        ('running', ['run_hardaware_guarded_soft5_admission.py']),
        ('auditing', ['audit_hardaware_guarded_soft5_admission.py']),
        ('reviewing', ['review_seeded_and_hardaware_admission.py', 'hardaware'])]:
        with (ROOT/'results'/f'hardaware-training-{phase}.log').open('w', encoding='utf-8') as log:
            child = subprocess.Popen([sys.executable, str(ROOT/'scripts'/command[0]), *command[1:]], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            status.write_text(json.dumps({'phase': phase, 'child_pid': child.pid, 'updated_unix': time.time(), 'goal_achieved': False}, indent=2))
            code = child.wait()
            if code:
                status.write_text(json.dumps({'phase': 'failed', 'failed_stage': phase, 'exit_code': code, 'goal_achieved': False}, indent=2))
                raise RuntimeError(f'{phase} failed; see log')
    status.write_text(json.dumps({'phase': 'complete_audited_reviewed', 'goal_achieved': False, 'scope': 'training_only'}, indent=2))


if __name__ == '__main__': main()
