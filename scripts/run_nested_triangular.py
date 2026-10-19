"""Complete independent inner fuzzy group/LP fitting with hierarchical choice."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib,json
from pathlib import Path
import numpy as np
import run_nested_teacher_regularized as runner
from certified_frfs.nested_triangular_stage import fit_stage
from certified_frfs.triangular_classifier import triangular_memberships

runner.FOLDER=runner.ROOT/'results/nested-triangular'
runner.fit_stage=fit_stage
original_select=runner.select

def select(inner):
    _,full,trace=original_select(inner);family=[r for r in trace['options'] if r['C']==full['C']]
    best=max(r['score'] for r in family);eligible=[r for r in family if r['score']>=best-.005-1e-12]
    proposed=min(eligible,key=lambda r:(r['retained'],-r['score'],float(r['fraction'])))
    return proposed,full,{'options':trace['options'],'eligible':eligible,'tolerance':.005,'hierarchical':True}

def score(raw,labels,query,truth,stage,*,noise=False,seed=144):
    from certified_frfs.validated_pool import preprocess_fit
    _,B=preprocess_fit(raw,query);queries=[B]
    if noise:
        rng=np.random.default_rng(seed);queries += [np.clip(B+rng.uniform(-.01,.01,B.shape),0,1) for _ in range(3)]
    values={}
    for key,r in stage['policies'].items():
        W,b=np.array(r['fit']['coef']),np.array(r['fit']['intercept']);classes=np.array(stage['teacher_classes']);scores=[]
        for Q in queries:
            pred=classes[(np.einsum('ndk,qdk->nq',triangular_memberships(Q),W)+b).argmax(axis=1)]
            scores.append(float(np.mean([np.mean(pred[truth==c]==c) for c in np.unique(truth)])))
        values[key]={'clean':scores[0],'query_noise01':float(np.mean(scores[1:])) if noise else None,'retained':len(r['candidate']['features'])/stage['d']}
    return values

runner.select=select;runner.base.score=score

def run(name):
    runner.run(name);path=runner.FOLDER/f'{name}.json';p=json.loads(path.read_text())
    files=[Path(__file__),runner.ROOT/'research/nested-triangular-protocol.md',runner.ROOT/'src/certified_frfs/nested_triangular_stage.py',runner.ROOT/'src/certified_frfs/fuzzy_teacher_support.py',runner.ROOT/'src/certified_frfs/triangular_classifier.py',runner.ROOT/'src/certified_frfs/triangular_robust_refit.py']
    p['fingerprints'].update({str(f.relative_to(runner.ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    path.write_text(json.dumps(p,indent=2),encoding='utf-8')

if __name__=='__main__':
    runner.FOLDER.mkdir(exist_ok=True);names=[r['dataset'] for r in json.loads((runner.ROOT/'results/triangular-robust-refit/complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,n) for n in names]):f.result()
    runner.summarize(names)
