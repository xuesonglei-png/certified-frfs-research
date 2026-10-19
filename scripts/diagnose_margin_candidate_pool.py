"""Add fixed MAFRFS-style margin supports to archived certified candidates."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib,json
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.baselines import select_mafrfs_additive
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.query_probability import QueryProbabilityObjective

FOLDER=ROOT/'results/margin-candidate-pool-v2';CS=(.1,1.,10.)


def score(A,y,B,truth,fs,C):
    return float(balanced_accuracy_score(truth,LogisticRegression(C=C,max_iter=2000,random_state=0).fit(A[:,fs],y).predict(B[:,fs])))


def run(name):
    threadpool_limits(1);out=FOLDER/f'{name}.json'
    if out.exists():raise FileExistsError(out)
    X,y=load_dataset(name);prior=json.loads((ROOT/'results/long-capped-sparse'/f'{name}.json').read_text());rows=[]
    for index,s in enumerate(prior['splits']):
        ri,vi=np.array(s['reference_indices']),np.array(s['validation_indices']);raw,labels=X[ri],y[ri];d=X.shape[1];mass=np.sqrt(d);A,B=preprocess_fit(raw,X[vi])
        candidates=list(s['fit']['candidates'])
        for fraction in (.35,.5,.65,.8,1.):
            k=max(1,int(np.ceil(fraction*d)))
            for margin_type in ('local','global'):
                fs=select_mafrfs_additive(A,labels,k=k,sop=2,margin_type=margin_type)
                w=np.zeros(d);w[fs]=mass/len(fs)
                cert=weight_certificate(A,labels,fs,np.ones(d)/mass,w[fs])
                if cert['feasible'] and w.max()<=2*mass/d+1e-8:
                    candidates.append({'origin':'margin_'+margin_type,'features':fs,'weights':w.tolist(),
                        'certificate':cert,'objective':None})
        candidates=[c for i,c in enumerate(candidates) if c['certificate']['feasible'] and not any(c['features']==z['features'] and c['weights']==z['weights'] for z in candidates[:i])]
        folds=[(np.array(f['reference']),np.array(f['query'])) for f in QueryProbabilityObjective(raw,labels,seed=1701).splits]
        trace=[]
        for ci,c in enumerate(candidates):
            for C in CS:
                vals=[]
                for ti,qi in folds:
                    Ai,Bi=preprocess_fit(raw[ti],raw[qi]);vals.append(score(Ai,labels[ti],Bi,labels[qi],c['features'],C))
                trace.append({'candidate_index':ci,'C':C,'mean_clean':float(np.mean(vals)),'folds':vals,'size':len(c['features']),'origin':c['origin']})
        chosen=max(range(len(trace)),key=lambda j:(trace[j]['mean_clean'],-trace[j]['size'],-j));t=trace[chosen];c=candidates[t['candidate_index']]
        rng=np.random.default_rng(144+index);versions=[(A,B)]
        for _ in range(3):versions.append((np.clip(A+rng.uniform(-.01,.01,A.shape),0,1),np.clip(B+rng.uniform(-.01,.01,B.shape),0,1)))
        values=[score(R,labels,Q,y[vi],c['features'],t['C']) for R,Q in versions]
        rows.append({'reference_indices':ri.tolist(),'validation_indices':vi.tolist(),'candidates':candidates,'trace':trace,'chosen_index':chosen,'selected':c,
            'C':t['C'],'metrics':{'clean':values[0],'noise01':float(np.mean(values[1:]))},'margin_candidates':sum(z['origin'].startswith('margin_') for z in candidates)})
        print(name,index,'selected',c['origin'],len(c['features']),flush=True)
    paths=[Path(__file__),ROOT/'research/margin-candidate-pool-protocol.md',ROOT/'src/certified_frfs/baselines.py',ROOT/'data/processed'/f'{name}.npz',ROOT/'results/long-capped-sparse'/f'{name}.json']
    out.write_text(json.dumps({'dataset':name,'d':d,'splits':rows,'fingerprints':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}},indent=2),encoding='utf-8')


def summarize(names):
    rows=[]
    for name in names:
        p=json.loads((FOLDER/f'{name}.json').read_text());s=p['splits'];rows.append({'dataset':name,'clean':float(np.mean([r['metrics']['clean'] for r in s])),
            'removed':float(np.mean([1-len(r['selected']['features'])/p['d'] for r in s])),'margin_selected':sum(r['selected']['origin'].startswith('margin_') for r in s)})
    controls=json.loads((ROOT/'results/nested-classifier-pipelines/complete.json').read_text())
    prior={r['dataset']:r for r in controls['rows']}
    comparisons={m:{'gain':float(np.mean([r['clean']-prior[r['dataset']]['means']['logistic'][m]['clean'] for r in rows])),'wins':sum(r['clean']>prior[r['dataset']]['means']['logistic'][m]['clean']+1e-12 for r in rows)} for m in ('full','fgs2023','sfrip2025','proposed_uniform')}
    removed=float(np.mean([r['removed'] for r in rows]))
    result={'rows':rows,'mean_clean':float(np.mean([r['clean'] for r in rows])),'mean_removed':removed,
        'gate_passed':removed>=.1 and comparisons['full']['gain']>=-.005 and all(comparisons[m]['gain']>=.005 and comparisons[m]['wins']>=10 for m in ('fgs2023','sfrip2025')),
        'margin_selected':sum(r['margin_selected'] for r in rows),'comparisons':comparisons,'scope':'all15_all44_margin_candidate_exposed_development_not_confirmation'}
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    names=[r['dataset'] for r in json.loads((ROOT/'results/long-capped-sparse/complete.json').read_text())['rows']];FOLDER.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,n) for n in names]):f.result()
    summarize(names)
