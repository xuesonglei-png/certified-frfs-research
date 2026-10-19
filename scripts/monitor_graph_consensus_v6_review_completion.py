"""Bounded checkpoint monitor; run the independent summary after all six finish.

This monitor neither starts nor duplicates model fitting. It watches the
already running sequential continuation and only summarizes completed artifacts.
"""
import json,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
NAMES=('banknote','glass','heart_statlog','ionosphere','seeds','wdbc')
STATUS=ROOT/'research/graph-consensus-inner-v6-review-progress.json'
deadline=time.monotonic()+6*60*60

while time.monotonic()<deadline:
    progress=[]
    for name in NAMES:
        p=ROOT/'results/graph-consensus-inner-v6'/(name+'.json')
        try:
            j=json.loads(p.read_text()) if p.exists() else {}
            progress.append({'dataset':name,'complete':j.get('complete',False),'panels':len(j.get('panels',[]))})
        except (json.JSONDecodeError,OSError):
            progress.append({'dataset':name,'complete':False,'panels':None,'checkpoint_temporarily_unreadable':True})
    STATUS.write_text(json.dumps({'progress':progress,'new_fitting_or_outer_scoring':False,'phase':'awaiting_sequential_trials'},indent=2),encoding='utf8')
    if all(p['complete'] for p in progress):
        summary=ROOT/'research/graph-consensus-inner-v6-reviewed-summary.json'
        if summary.exists():
            code=0
        else:
            result=subprocess.run([sys.executable,str(ROOT/'scripts/summarize_graph_consensus_inner_v6_review.py')],cwd=ROOT,capture_output=True,text=True)
            (ROOT/'research/graph-consensus-inner-v6-summary-run.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf8')
            code=result.returncode
        STATUS.write_text(json.dumps({'progress':progress,'new_fitting_or_outer_scoring':False,'phase':'summary_complete' if code==0 else 'summary_failed','summary_exit_code':code},indent=2),encoding='utf8')
        print('v6 review summary exit',code,flush=True)
        sys.exit(code)
    continuation=ROOT/'research/graph-consensus-inner-v6-continuation-status.json'
    if continuation.exists():
        state=json.loads(continuation.read_text())
        if len(state.get('records',[]))==5:
            STATUS.write_text(json.dumps({'progress':progress,'new_fitting_or_outer_scoring':False,'phase':'continuation_finished_with_incomplete_or_failed_trial','continuation':state},indent=2),encoding='utf8')
            print('continuation finished with incomplete trial; retained failures',flush=True)
            sys.exit(2)
    time.sleep(20)
STATUS.write_text(json.dumps({'progress':progress,'phase':'bounded_monitor_timeout','new_fitting_or_outer_scoring':False},indent=2),encoding='utf8')
sys.exit(3)
