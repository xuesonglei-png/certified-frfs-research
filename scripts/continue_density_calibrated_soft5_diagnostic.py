"""Run complete fixed cohort, independent audit, then inner-only review."""
import json
import subprocess
import sys
import time
from run_density_calibrated_soft5_diagnostic import ROOT,FOLDER,digest


def main():
    status=ROOT/'results/density-calibrated-soft5-diagnostic-chain-status.json'
    def update(phase,**kw):
        status.write_text(json.dumps({'phase':phase,'updated_unix':time.time(),
            'goal_achieved':False,**kw},indent=2),encoding='utf-8')
    try:
        frozen=json.loads((ROOT/'research/density-calibrated-soft5-diagnostic-source-freeze.json').read_text())
        for rel,sha in frozen['fingerprints'].items():assert digest(ROOT/rel)==sha,rel
        for phase,script in [('running','run_density_calibrated_soft5_diagnostic.py'),
            ('auditing','audit_density_calibrated_soft5_diagnostic.py'),
            ('reviewing','review_density_calibrated_soft5_diagnostic.py')]:
            with (ROOT/f'results/density-calibrated-soft5-diagnostic-{phase}.log').open('w',encoding='utf-8') as log:
                child=subprocess.Popen([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                update(phase,child_pid=child.pid)
                if child.wait():raise RuntimeError(f'{phase} failed; see retained log')
        update('complete_inner_audited_reviewed')
    except Exception as error:
        update('failed',error=str(error));raise


if __name__=='__main__':main()
