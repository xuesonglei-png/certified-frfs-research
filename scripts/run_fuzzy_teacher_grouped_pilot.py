"""Fuzzy teacher score-group supports with exact shared-source evaluation."""
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
from certified_frfs.fuzzy_teacher_support import fuzzy_score_supports
from certified_frfs.feasible_transport import weight_certificate
from review_box_iterations import paired

FOLDER=ROOT/'results/fuzzy-teacher-grouped'

def train(A,y,fs,C):
    Z=triangular_memberships(A[:,fs]).reshape(len(A),-1);model=LogisticRegression(C=C,max_iter=2000,random_state=0).fit(Z,y)
    coef,b,classes=linear_parameters(model);W=np.zeros((len(classes),A.shape[1],3));W[:,fs]=coef.reshape(len(classes),len(fs),3)
    return {'coef':W.tolist(),'intercept':b.tolist(),'classes':classes.tolist()}

def score(B,truth,fit):
    W=np.array(fit['coef']);b=np.array(fit['intercept']);classes=np.array(fit['classes']);pred=classes[(np.einsum('ndk,qdk->nq',triangular_memberships(B),W)+b).argmax(axis=1)]
    return float(np.mean([np.mean(pred[truth==c]==c) for c in np.unique(truth)]))

def run(name):
    threadpool_limits(1);out=FOLDER/f'{name}.json'
    if out.exists():raise FileExistsError(out)
    source=json.loads((ROOT/'results/common-fuzzy-classifier'/f'{name}.json').read_text());X,y=load_dataset(name);rows=[]
    for i,s in enumerate(source['splits']):
        ri,vi=np.array(s['reference_indices']),np.array(s['validation_indices']);A,B=preprocess_fit(X[ri],X[vi]);
        full=s['results']['full'];C=full['C'];teacher= train(A,y[ri],list(range(X.shape[1])),C);W=np.array(teacher['coef']);b=np.array(teacher['intercept']);classes=np.array(teacher['classes'])
        generated=fuzzy_score_supports(A,y[ri],W,b);trials=[]
        for c in generated:
            fit=train(A,y[ri],c['features'],C);margin=triangular_box_margins(A,y[ri],np.array(fit['coef']),np.array(fit['intercept']),classes)
            original=triangular_box_margins(A,y[ri],W,b,classes);protected=original>1e-10;cert=bool(np.all(margin[protected]>=.5*original[protected]-1e-8))
            trials.append({'candidate':c,'fit':fit,'certificate':{'feasible':cert,'protected_count':int(protected.sum()),'retained':int(np.sum((margin>1e-10)&protected)),'margins':margin.tolist()}})
        fullmargin=triangular_box_margins(A,y[ri],W,b,classes);feasible=[t for t in trials if t['certificate']['feasible']];selected=min(feasible,key=lambda t:(len(t['candidate']['features']),t['candidate']['reconstruction_error'])) if feasible else {'candidate':{'features':list(range(X.shape[1]))},'fit':teacher,'certificate':{'feasible':True,'fallback':True}}
        rng=np.random.default_rng(144+i);queries=[B]+[np.clip(B+rng.uniform(-.01,.01,B.shape),0,1) for _ in range(3)]
        scores=[score(Q,y[vi],selected['fit']) for Q in queries];fullscores=[score(Q,y[vi],teacher) for Q in queries]
        rows.append({'reference_indices':ri.tolist(),'validation_indices':vi.tolist(),'C':C,'teacher':teacher,'candidates':generated,'trials':trials,'selected':selected,'fallback':not feasible,
            'metrics':{'clean':scores[0],'noise':float(np.mean(scores[1:]))},'full_metrics':{'clean':fullscores[0],'noise':float(np.mean(fullscores[1:]))},
            'full_certified':int(np.sum(fullmargin>1e-10)),'selected_removed':1-len(selected['candidate']['features'])/X.shape[1]})
        print(name,i,'C',C,'support',len(selected['candidate']['features']),'BA',scores[0],flush=True)
    paths=[Path(__file__),ROOT/'research/fuzzy-teacher-grouped-protocol.md',ROOT/'src/certified_frfs/fuzzy_teacher_support.py',ROOT/'src/certified_frfs/triangular_classifier.py',ROOT/'src/certified_frfs/validated_pool.py',ROOT/'data/processed'/f'{name}.npz']
    out.write_text(json.dumps({'dataset':name,'d':X.shape[1],'splits':rows,'fingerprints':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}},indent=2),encoding='utf-8')

def summarize(names):
    rows=[]
    for name in names:
        p=json.loads((FOLDER/f'{name}.json').read_text());s=p['splits'];rows.append({'dataset':name,'clean':float(np.mean([r['metrics']['clean'] for r in s])),'full':float(np.mean([r['full_metrics']['clean'] for r in s])),'removed':float(np.mean([r['selected_removed'] for r in s])),'fallbacks':sum(r['fallback'] for r in s)})
    result={'rows':rows,'mean_clean':float(np.mean([r['clean'] for r in rows])),'mean_full':float(np.mean([r['full'] for r in rows])),'mean_removed':float(np.mean([r['removed'] for r in rows])),'vs_full':paired([r['clean']-r['full'] for r in rows]),'scope':'fuzzy_teacher_grouped_support_fixed_C_exposed_pilot_not_complete_nested_confirmation'}
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(result)

if __name__=='__main__':
    FOLDER.mkdir(exist_ok=True);names=[r['dataset'] for r in json.loads((ROOT/'results/common-fuzzy-classifier/complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,n) for n in names]):f.result()
    summarize(names)
