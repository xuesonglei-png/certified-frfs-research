"""Complete native comparator run and independent audit without manuscript edits."""
import json
import subprocess
import sys
import time
from run_dsnfrs_formula_native_pilot import ROOT


def main():
    status = ROOT/'results/dsnfrs-formula-native-chain-status.json'
    for phase, script in [('running', 'run_dsnfrs_formula_native_pilot.py'), ('auditing', 'audit_dsnfrs_formula_native_pilot.py')]:
        with (ROOT/'results'/f'dsnfrs-formula-native-{phase}.log').open('w',encoding='utf-8') as log:
            child = subprocess.Popen([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            status.write_text(json.dumps({'phase':phase,'child_pid':child.pid,'updated_unix':time.time(),'goal_achieved':False},indent=2))
            if child.wait():
                status.write_text(json.dumps({'phase':'failed','failed_stage':phase,'goal_achieved':False},indent=2))
                raise RuntimeError(f'{phase} failed; see log')
    status.write_text(json.dumps({'phase':'complete_audited','goal_achieved':False,'scope':'formula_controlled_baseline_not_author_reproduction'},indent=2))


if __name__ == '__main__': main()
