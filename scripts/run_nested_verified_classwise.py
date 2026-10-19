"""Complete nested hierarchical teacher/support choice with classwise90 MILP."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib,json
from pathlib import Path
import run_nested_teacher_regularized as runner
import certified_frfs.nested_teacher_regularized as stages
from certified_frfs.verified_classwise_box_linear_refit import fit_verified_classwise_box_linear

stages.fit_distilled_box_linear=fit_verified_classwise_box_linear
runner.FOLDER=runner.ROOT/'results/nested-verified-classwise'
original_select=runner.select

def hierarchical_select(inner):
    _,full,trace=original_select(inner)
    options=[r for r in trace['options'] if r['C']==full['C']]
    best=max(r['score'] for r in options)
    eligible=[r for r in options if r['score']>=best-.005-1e-12]
    chosen=min(eligible,key=lambda r:(r['retained'],-r['score'],float(r['fraction'])))
    return chosen,full,{'options':trace['options'],'eligible':eligible,'tolerance':.005,'hierarchical':True}

runner.select=hierarchical_select

def run(name):
    runner.run(name)
    path=runner.FOLDER/f'{name}.json';p=json.loads(path.read_text())
    files=[Path(__file__),runner.ROOT/'research/nested-verified-classwise-protocol.md',runner.ROOT/'src/certified_frfs/verified_classwise_box_linear_refit.py']
    p['fingerprints'].update({str(f.relative_to(runner.ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    path.write_text(json.dumps(p,indent=2),encoding='utf-8')

if __name__=='__main__':
    runner.FOLDER.mkdir(exist_ok=True);names=[r['dataset'] for r in json.loads((runner.ROOT/'results/verified-classwise-box-pilot/complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,n) for n in names]):f.result()
    runner.summarize(names)
