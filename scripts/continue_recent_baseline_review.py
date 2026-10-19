"""Incremental audited recent-baseline reports as existing cohorts finish."""
import json
import subprocess
import sys
import time
from run_dsnfrs_formula_native_pilot import ROOT, digest
from continue_joint_guarded_soft5_knn_pilot import wait_verified


def main():
    status = ROOT/'results/recent-baseline-review-chain-status.json'
    def update(phase, **values):
        status.write_text(json.dumps({'phase':phase, 'updated_unix':time.time(),
                                     'goal_achieved':False, **values}, indent=2), encoding='utf-8')
    def review(label):
        with (ROOT/'results'/f'recent-baseline-review-{label}.log').open('w',encoding='utf-8') as stream:
            child = subprocess.Popen([sys.executable,str(ROOT/'scripts/review_dsnfrs_formula_comparison.py')],
                                     cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            update('reviewing_'+label, child_pid=child.pid)
            if child.wait(): raise RuntimeError(f'{label} review failed; see log')
        report = ROOT/'research/dsnfrs-formula-comparison-review.json'
        snapshot = ROOT/'research'/f'dsnfrs-formula-comparison-review-{label}.json'
        assert not snapshot.exists()
        snapshot.write_bytes(report.read_bytes())
        update('reviewed_'+label, report_sha256=digest(snapshot))
    try:
        for name, pid in [('dsnfrs-formula-native-pilot',50500), ('simplex-soft5-knn-pilot',47888)]:
            update('waiting_'+name, prerequisite_pid=pid)
            wait_verified(ROOT/'results'/name,pid)
        review('native-cap1')
        for label,name,pid in [('expanded-cap1','dsnfrs-formula-expanded-pilot',49216),
                               ('joint','joint-guarded-soft5-knn-pilot',48156),
                               ('hardaware','hardaware-soft5-knn-pilot',50996)]:
            update('waiting_'+name,prerequisite_pid=pid)
            wait_verified(ROOT/'results'/name,pid)
            review(label)
        update('all_available_cohorts_reviewed', scope='audited_development_not_confirmed_superiority')
    except Exception as error:
        update('failed',error=str(error))
        raise


if __name__ == '__main__': main()
