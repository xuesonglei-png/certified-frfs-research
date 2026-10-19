"""Preserve V1 adapter failure and audit unchanged experiment with V2."""
import json
import subprocess
import sys
import time
from pathlib import Path
from run_converged_geometric_benchmark import ROOT,FOLDER,digest

STATUS=ROOT/'results/converged-geometric-benchmark-audit-v2-status.json'
ORIGINAL=ROOT/'results/converged-geometric-benchmark-chain-status.json'


def status(phase,**values):
    STATUS.write_text(json.dumps({'phase':phase,'updated_unix':time.time(),'goal_achieved':False,**values},indent=2),encoding='utf-8')


def main():
    status('waiting_for_original_chain_terminal_state')
    while True:
        original=json.loads(ORIGINAL.read_text())
        if original['phase'] in ('failed','complete_audit_ready_for_review'):break
        time.sleep(3)
    repair=json.loads((ROOT/'research/converged-geometric-auditor-repair.json').read_text())
    script=ROOT/'scripts/audit_converged_geometric_benchmark_v2.py'
    assert digest(script)==repair['replacement_auditor_sha256']
    assert digest(ROOT/'scripts/audit_converged_geometric_benchmark.py')==repair['original_auditor_sha256']
    if original['phase']=='failed':
        assert original['error'].startswith('audit_converged_geometric_benchmark.py failed'),original
    summary=json.loads((FOLDER/'complete.json').read_text());assert summary['counts']['datasets']==15
    log=ROOT/'results/converged-geometric-benchmark-audit-v2.log'
    with log.open('a',encoding='utf-8') as stream:
        child=subprocess.Popen([sys.executable,str(script)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        status('auditing',child_pid=child.pid,log=log.relative_to(ROOT).as_posix(),original_chain_state_preserved=original)
        if child.wait():
            status('failed',error='V2 independent audit failed',original_chain_state_preserved=original)
            raise RuntimeError('V2 independent audit failed')
    reviewlog=ROOT/'results/converged-geometric-benchmark-review.log'
    with reviewlog.open('a',encoding='utf-8') as stream:
        child=subprocess.Popen([sys.executable,str(ROOT/'scripts/review_converged_geometric_benchmark.py')],cwd=ROOT,
                               stdout=stream,stderr=subprocess.STDOUT)
        status('reviewing',child_pid=child.pid,log=reviewlog.relative_to(ROOT).as_posix())
        if child.wait():
            status('failed',error='Review of V2 audited cohort failed');raise RuntimeError('Audited review failed')
    status('complete_audited_reviewed',original_chain_state_preserved=original,
           pilot_gate_passed=summary['pilot_gate_passed'],
           audit_sha256=digest(FOLDER/'complete-audit.json'),auditor_repair_sha256=digest(ROOT/'research/converged-geometric-auditor-repair.json'))


if __name__=='__main__':main()
