"""Apply fixed-linear margin retention to archived certified candidates."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.linear_retention import linear_parameters,linear_retention_certificate,projected_parameters

FOLDER=ROOT/'results/linear-retention';CS=(.1,1.,10.)

def score(A,y,B,truth,fs,C):
    model=LogisticRegression(C=C,max_iter=2000,random_state=0).fit(A[:,fs],y)
    return float(balanced_accuracy_score(truth,model.predict(B[:,fs])))

def run(name):
    threadpool_limits(1);out=FOLDER/f'{name}.json'
    if out.exists():raise FileExistsError(out)
    X,y=load_dataset(name);prior=json.loads((ROOT/'results/long-capped-sparse'/f'{name}.json').read_text());nested=json.loads((ROOT/'results/nested-classifier-pipelines'/f'{name}.json').read_text());rows=[]
    for index,s in enumerate(prior['splits']):
        ri,vi=np.array(s['reference_indices']),np.array(s['validation_indices']);raw,labels=X[ri],y[ri];A,B=preprocess_fit(raw,X[vi]);d=X.shape[1];mass=np.sqrt(d)
        C=float(nested['splits'][index]['trace']['logistic']['full'][nested['splits'][index]['results']['logistic']['full']['chosen_index']]['C'])
        teacher=LogisticRegression(C=C,max_iter=2000,random_state=0).fit(A,labels);W,b,classes=linear_parameters(teacher);center=A.mean(axis=0)
        candidates=[c for c in s['fit']['candidates'] if c['certificate']['feasible']]
        audited=[]
        for c in candidates:
            cert=linear_retention_certificate(A,labels,W,b,classes,c['features'],center,margin_ratio=.5)
            audited.append({'candidate':c,'linear_certificate':cert})
        feasible=[z for z in audited if z['linear_certificate']['feasible']]
        if not feasible:raise RuntimeError(f'no linear candidate {name} {index}')
        selected=min(feasible,key=lambda z:(len(z['candidate']['features']),z['candidate']['objective'] if z['candidate']['objective'] is not None else 1e9))['candidate']
        Rw,Rb=projected_parameters(W,b,selected['features'],center)
        rng=np.random.default_rng(144+index);versions=[(A,B)]
        for _ in range(3):versions.append((np.clip(A+rng.uniform(-.01,.01,A.shape),0,1),np.clip(B+rng.uniform(-.01,.01,B.shape),0,1)))
        values=[score(R,labels,Q,y[vi],selected['features'],C) for R,Q in versions]
        projected=[float(balanced_accuracy_score(y[vi],classes[(Q@Rw.T+Rb).argmax(axis=1)])) for _,Q in versions]
        rows.append({'reference_indices':ri.tolist(),'validation_indices':vi.tolist(),'teacher_C':C,'teacher_classes':classes.tolist(),'teacher_coef':W.tolist(),'teacher_intercept':b.tolist(),'center':center.tolist(),
            'candidates':audited,'selected':selected,'refit_metrics':{'clean':values[0],'noise01':float(np.mean(values[1:]))},
            'metrics':{'clean':projected[0],'noise01':float(np.mean(projected[1:]))}})
        print(name,index,'support',len(selected['features']),'clean',values[0],flush=True)
    paths=[Path(__file__),ROOT/'research/linear-retention-protocol.md',ROOT/'src/certified_frfs/linear_retention.py',ROOT/'results/long-capped-sparse'/f'{name}.json',ROOT/'results/nested-classifier-pipelines'/f'{name}.json',ROOT/'data/processed'/f'{name}.npz']
    out.write_text(json.dumps({'dataset':name,'d':d,'splits':rows,'fingerprints':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}},indent=2),encoding='utf-8')

def summarize(names):
    rows=[];nested=json.loads((ROOT/'results/nested-classifier-pipelines/complete.json').read_text())
    for name in names:
        p=json.loads((FOLDER/f'{name}.json').read_text());s=p['splits'];rows.append({'dataset':name,'clean':float(np.mean([r['metrics']['clean'] for r in s])),'removed':float(np.mean([1-len(r['selected']['features'])/p['d'] for r in s])),'linear_feasible_candidates':sum(sum(z['linear_certificate']['feasible'] for z in r['candidates']) for r in s)})
    prior={r['dataset']:r for r in nested['rows']}
    comparisons={m:{'gain':float(np.mean([r['clean']-prior[r['dataset']]['means']['logistic'][m]['clean'] for r in rows])),'wins':sum(r['clean']>prior[r['dataset']]['means']['logistic'][m]['clean']+1e-12 for r in rows)} for m in ('full','proposed_uniform','fgs2023','sfrip2025')}
    result={'rows':rows,'mean_clean':float(np.mean([r['clean'] for r in rows])),'mean_removed':float(np.mean([r['removed'] for r in rows])),'comparisons':comparisons,'scope':'all15_all44_fixed_logistic_margin_retention_exposed_development_not_confirmation'}
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    names=[r['dataset'] for r in json.loads((ROOT/'results/long-capped-sparse/complete.json').read_text())['rows']];FOLDER.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,n) for n in names]):f.result()
    summarize(names)
