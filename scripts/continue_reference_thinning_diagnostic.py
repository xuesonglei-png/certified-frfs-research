"""Run the registered training diagnosis and require its independent replay."""
import json
import subprocess
import sys
import time
from run_reference_thinning_diagnostic import ROOT,FOLDER,digest


def main():
    status=ROOT/'results/reference-thinning-diagnostic-chain-status.json'
    def update(phase,**kwargs):
        status.write_text(json.dumps({'phase':phase,'updated_unix':time.time(),
            'goal_achieved':False,**kwargs},indent=2),encoding='utf-8')
    try:
        frozen=json.loads((ROOT/'research/reference-thinning-diagnostic-source-freeze.json').read_text())
        for rel,sha in frozen['fingerprints'].items():assert digest(ROOT/rel)==sha
        for phase,script in [('running','run_reference_thinning_diagnostic.py'),('auditing','audit_reference_thinning_diagnostic.py')]:
            with (ROOT/'results'/f'reference-thinning-diagnostic-{phase}.log').open('w',encoding='utf-8') as log:
                child=subprocess.Popen([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                update(phase,child_pid=child.pid)
                if child.wait(): raise RuntimeError(f'{phase} failed; inspect retained log')
        result=json.loads((FOLDER/'complete.json').read_text())
        audit=json.loads((FOLDER/'complete-audit.json').read_text());assert audit['passed']
        assert digest(FOLDER/'complete.json')==audit['summary_sha256']
        update('complete_audited_training_diagnostic',counts=result['counts'],audit_sha256=digest(FOLDER/'complete-audit.json'))
    except Exception as error:
        update('failed',error=str(error));raise


if __name__=='__main__':main()
