"""Fixed fuzzy additive teacher capacity and uncertainty-coupling diagnostic."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.linear_retention import linear_parameters
from certified_frfs.triangular_classifier import triangular_memberships,triangular_box_margins
from review_box_iterations import paired

FOLDER=ROOT/'results/triangular-classifier-diagnostic';CS=(.1,1.,10.)

def fit(A,y,C):
    D=triangular_memberships(A).reshape(len(A),-1)
    return LogisticRegression(C=C,max_iter=2000,random_state=0).fit(D,y)

def ba(truth,pred):return float(np.mean([np.mean(pred[truth==c]==c) for c in np.unique(truth)]))

def run(name):
    threadpool_limits(1);out=FOLDER/f'{name}.json'
    if out.exists():raise FileExistsError(out)
    paths=[ROOT/'results'/f/f'{name}.json' for f in ('nested-verified-classwise','common-fold-recent')]
    source,baseline=[json.loads(p.read_text()) for p in paths];X,y=load_dataset(name);rows=[]
    for i,s in enumerate(source['splits']):
        ri,vi=np.array(s['reference_indices']),np.array(s['validation_indices']);raw,labels=X[ri],y[ri];trace={str(C):[] for C in CS}
        for inner in s['inner']:
            a,b=np.array(inner['reference_indices']),np.array(inner['validation_indices']);A,B=preprocess_fit(raw[a],raw[b]);query=triangular_memberships(B).reshape(len(B),-1)
            for C in CS:trace[str(C)].append(ba(labels[b],fit(A,labels[a],C).predict(query)))
        chosen=min(CS,key=lambda C:(-np.mean(trace[str(C)]),C));A,B=preprocess_fit(raw,X[vi]);model=fit(A,labels,chosen)
        W,b,classes=linear_parameters(model);W=W.reshape(len(classes),X.shape[1],3)
        margins=triangular_box_margins(B,y[vi],W,b,classes);relaxed=triangular_box_margins(B,y[vi],W,b,classes,independent_memberships=True)
        assert np.all(relaxed<=margins+1e-10)
        cert=lambda m:float(np.mean([np.mean(m[y[vi]==c]>1e-10) for c in np.unique(y[vi])]))
        clean=ba(y[vi],model.predict(triangular_memberships(B).reshape(len(B),-1)))
        rows.append({'reference_indices':ri.tolist(),'validation_indices':vi.tolist(),'inner_trace':trace,'C':chosen,'coef':W.tolist(),'intercept':b.tolist(),'classes':classes.tolist(),
            'clean':clean,'raw_full_clean':baseline['splits'][i]['results']['full']['metrics']['clean'],
            'coupled_certified':cert(margins),'independent_certified':cert(relaxed),'coupled_margins':margins.tolist(),'independent_margins':relaxed.tolist()})
        print(name,i,'fuzzy teacher clean',clean,'C',chosen,flush=True)
    paths += [Path(__file__),ROOT/'research/triangular-classifier-protocol.md',ROOT/'src/certified_frfs/triangular_classifier.py',ROOT/'src/certified_frfs/validated_pool.py',ROOT/'data/processed'/f'{name}.npz']
    out.write_text(json.dumps({'dataset':name,'splits':rows,'fingerprints':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}},indent=2),encoding='utf-8')

def summarize(names):
    rows=[]
    for name in names:
        p=json.loads((FOLDER/f'{name}.json').read_text());rows.append({'dataset':name,**{k:float(np.mean([s[k] for s in p['splits']])) for k in ('clean','raw_full_clean','coupled_certified','independent_certified')}})
    result={'rows':rows,'means':{k:float(np.mean([r[k] for r in rows])) for k in ('clean','raw_full_clean','coupled_certified','independent_certified')},
        'capacity_gain':paired([r['clean']-r['raw_full_clean'] for r in rows]),'coupling_coverage_gain':paired([r['coupled_certified']-r['independent_certified'] for r in rows]),
        'scope':'fixed_fuzzy_granulation_full_capacity_and_exact_shared_source_test_certificate_diagnostic_no_reduction_or_confirmation'}
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print({k:v for k,v in result.items() if k!='rows'},flush=True)

if __name__=='__main__':
    FOLDER.mkdir(exist_ok=True);names=[r['dataset'] for r in json.loads((ROOT/'results/nested-verified-classwise/complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,n) for n in names]):f.result()
    summarize(names)
