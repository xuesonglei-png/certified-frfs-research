"""Run, independently audit, and preserve failure status without promotion."""
import json
import subprocess
import sys
import time
import traceback
from run_converged_geometric_benchmark import ROOT

STATUS=ROOT/'results/converged-geometric-benchmark-chain-status.json'


def status(phase,**values):
    STATUS.write_text(json.dumps({'phase':phase,'updated_unix':time.time(),'goal_achieved':False,**values},indent=2),encoding='utf-8')


def main():
    try:
        for phase,script in [('running','run_converged_geometric_benchmark.py'),('auditing','audit_converged_geometric_benchmark.py')]:
            log=ROOT/'results'/f'converged-geometric-benchmark-{phase}.log'
            with log.open('a',encoding='utf-8') as stream:
                child=subprocess.Popen([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                status(phase,child_pid=child.pid,log=log.relative_to(ROOT).as_posix())
                if child.wait():raise RuntimeError(f'{script} failed; see {log.name}')
        status('complete_audit_ready_for_review')
    except Exception as error:
        status('failed',error=str(error));traceback.print_exc();raise


if __name__=='__main__':main()
