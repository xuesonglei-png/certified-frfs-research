"""Teacher-aligned candidate generation and independent fuzzy certification."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib,json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import diagnose_robust_linear_refit as search
from compare_matched_robust_controls import metrics
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.teacher_support import reconstruction_supports
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.distilled_box_linear_refit import fit_distilled_box_linear

ROOT=search.ROOT;FOLDER=ROOT/'results/teacher-support-refit'

def run(name):
    threadpool_limits(1);out=FOLDER/f'{name}.json'
    if out.exists():raise FileExistsError(out)
    priorpath=ROOT/'results/distilled-box-linear-refit'/f'{name}.json';teacherpath=ROOT/'results/linear-retention'/f'{name}.json'
    p=json.loads(priorpath.read_text());teachers=json.loads(teacherpath.read_text());X,y=load_dataset(name);rows=[]
    for i,s in enumerate(p['splits']):
        ri,vi=np.array(s['reference_indices']),np.array(s['validation_indices']);A,B=preprocess_fit(X[ri],X[vi]);t=teachers['splits'][i]
        W,b,classes=np.array(t['teacher_coef']),np.array(t['teacher_intercept']),np.array(t['teacher_classes'])
        generated=reconstruction_supports(A,y[ri],W,b);trials=list(s['trials']);seen={tuple(sorted(r['candidate']['features'])) for r in trials}
        for c in generated:
            fs=c['features'];c['certificate']=weight_certificate(A,y[ri],fs,np.ones(p['d'])/np.sqrt(p['d']),np.array(c['weights'])[fs])
            if c['certificate']['feasible'] and tuple(sorted(fs)) not in seen:
                trials.append({'candidate':c,'fit':fit_distilled_box_linear(A,y[ri],W,b,classes,fs)})
                seen.add(tuple(sorted(fs)))
        feasible=[r for r in trials if r['fit']['feasible']];selected=min(feasible,key=lambda r:(len(r['candidate']['features']),r['fit']['objective']))
        rng=np.random.default_rng(144+i);queries=[B]+[np.clip(B+rng.uniform(-.01,.01,B.shape),0,1) for _ in range(3)]
        value=metrics(queries,y[vi],classes,selected['fit'])
        rows.append({'reference_indices':ri.tolist(),'validation_indices':vi.tolist(),'trials':trials,'selected':selected,'generated':generated,'fallback':False,'metrics':value})
        print(name,i,'new feasible',sum(c['certificate']['feasible'] for c in generated),'size',len(selected['candidate']['features']),flush=True)
    files=[Path(__file__),ROOT/'research/teacher-support-protocol.md',ROOT/'src/certified_frfs/teacher_support.py',ROOT/'src/certified_frfs/distilled_box_linear_refit.py',ROOT/'src/certified_frfs/feasible_transport.py',priorpath,teacherpath,ROOT/'data/processed'/f'{name}.npz']
    out.write_text(json.dumps({'dataset':name,'d':p['d'],'splits':rows,'fingerprints':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}},indent=2),encoding='utf-8')

if __name__=='__main__':
    FOLDER.mkdir(exist_ok=True);names=[r['dataset'] for r in json.loads((ROOT/'results/distilled-box-linear-refit/complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,n) for n in names]):f.result()
    search.FOLDER=FOLDER;search.summarize(names)
