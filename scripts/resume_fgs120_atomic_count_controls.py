"""Resume the interrupted registered cohort without overwriting prior logs."""
import json
import subprocess
import sys
import time
from run_fgs120_atomic_count_controls import ROOT,SOURCE,FOLDER,digest
from continue_soft_topk_knn_pilot import verify


def main():
    frozen=json.loads((ROOT/'research/fgs120-atomic-count-controls-source-freeze.json').read_text())
    for rel,sha in frozen['fingerprints'].items():assert digest(ROOT/rel)==sha,rel
    verify(SOURCE)
    status=ROOT/'results/fgs120-atomic-count-controls-chain-status.json'
    stamp=str(int(time.time()))
    if status.exists():
        (ROOT/f'results/fgs120-atomic-count-controls-interrupted-status-{stamp}.json').write_bytes(status.read_bytes())
    def update(phase,**values):
        status.write_text(json.dumps({'phase':phase,'updated_unix':time.time(),
            'goal_achieved':False,'resume_stamp':stamp,**values},indent=2),encoding='utf-8')
    try:
        for phase,script in [('running','run_fgs120_atomic_count_controls.py'),('auditing','audit_fgs120_atomic_count_controls.py')]:
            logfile=ROOT/f'results/fgs120-atomic-count-controls-{phase}-resume-{stamp}.log'
            with logfile.open('w',encoding='utf-8') as log:
                child=subprocess.Popen([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT,
                    stdout=log,stderr=subprocess.STDOUT)
                update(phase,child_pid=child.pid,log=logfile.relative_to(ROOT).as_posix())
                if child.wait():raise RuntimeError(f'{phase} failed; see retained log')
        verify(FOLDER);update('complete_audited',audit_sha256=digest(FOLDER/'complete-audit.json'))
    except Exception as error:
        update('failed',error=str(error));raise


if __name__=='__main__':main()
