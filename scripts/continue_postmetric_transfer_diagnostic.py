"""Run inner-only diagnosis once the full independent audit is available."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT/'results/postmetric-transfer-diagnostic-chain-status.json'
SCRIPT = ROOT/'scripts/diagnose_postmetric_pruning_transfer.py'


def main():
    frozen = hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    deadline = time.monotonic()+24*3600
    while time.monotonic() < deadline:
        assert hashlib.sha256(SCRIPT.read_bytes()).hexdigest() == frozen
        audit = ROOT/'results/postmetric-pruning-classifier/complete-audit.json'
        if audit.exists():
            record = json.loads(audit.read_text())
            if record.get('passed'):
                break
        STATUS.write_text(json.dumps({'phase':'waiting_complete_independent_audit',
            'goal_achieved':False,'diagnostic_script_sha256':frozen,'updated_unix':time.time()},indent=2),encoding='utf-8')
        time.sleep(45)
    else:
        raise TimeoutError('complete independent audit unavailable after 24h; no scientific result generated')
    with (ROOT/'results/postmetric-transfer-diagnostic.log').open('w',encoding='utf-8') as log:
        child = subprocess.run([sys.executable,str(SCRIPT)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    STATUS.write_text(json.dumps({'phase':'complete' if child.returncode==0 else 'failed',
        'exit_code':child.returncode,'goal_achieved':False,'diagnostic_script_sha256':frozen,
        'updated_unix':time.time()},indent=2),encoding='utf-8')
    if child.returncode:
        raise RuntimeError('inner transfer diagnostic failed; inspect retained log')


if __name__ == '__main__':
    main()
