from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from run_newdata_compact import DATA, FOLDER as OLD_FOLDER, SOURCES
from certified_frfs.validated_pool import preprocess_fit

FOLDER=ROOT/'results/newdata-logistic-transfer-v2'; CS=(.1,1.,10.)


def score(A,y,B,truth,C):
    if A.shape[1] == 0:
        pred=DummyClassifier(strategy='prior').fit(np.zeros((len(y),1)),y).predict(np.zeros((len(B),1)))
    else:
        pred=LogisticRegression(C=C,max_iter=2000,random_state=0).fit(A,y).predict(B)
    return float(balanced_accuracy_score(truth,pred))


def fingerprints(name):
    paths=[Path(__file__),ROOT/'research/newdata-logistic-transfer-protocol.md',
           OLD_FOLDER/f'{name}.json',DATA/f'{name}.npz']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run(name):
    threadpool_limits(1); out=FOLDER/f'{name}.json'; hashes=fingerprints(name)
    if out.exists(): assert json.loads(out.read_text())['fingerprints']==hashes; return
    p=json.loads((OLD_FOLDER/f'{name}.json').read_text()); with_data=np.load(DATA/f'{name}.npz');X,y=with_data['X'],with_data['y'];rows=[]
    for fold,s in enumerate(p['splits']):
        tr,te=np.array(s['train_indices']),np.array(s['test_indices']);raw,labels=X[tr],y[tr]
        supports={'proposed':s['fit']['compact_selected']['features'],'full':list(range(p['d']))}
        supports.update({m:s['baselines'][m]['selection']['features'] for m in ('fgs2023','sfrip2025')})
        inner=s['fit']['training_splits'];choices={}
        for method,fs in supports.items():
            candidate=[]
            for C in CS:
                values=[]
                for split in inner:
                    ti,qi=np.array(split['reference']),np.array(split['query']);A,B=preprocess_fit(raw[ti],raw[qi])
                    values.append(score(A[:,fs],labels[ti],B[:,fs],labels[qi],C))
                candidate.append({'C':C,'inner_clean':float(np.mean(values)),'inner_folds':values})
            chosen=max(range(len(candidate)),key=lambda i:(candidate[i]['inner_clean'],-len(fs),-i));choices[method]={'candidates':candidate,'chosen_index':chosen}
        A,B=preprocess_fit(raw,X[te]);rng=np.random.default_rng(144+fold);versions=[(A,B)]
        for _ in range(3):versions.append((np.clip(A+rng.uniform(-.01,.01,A.shape),0,1),np.clip(B+rng.uniform(-.01,.01,B.shape),0,1)))
        scores={}
        for method,fs in supports.items():
            C=CS[choices[method]['chosen_index']];values=[score(R[:,fs],labels,Q[:,fs],y[te],C) for R,Q in versions]
            scores[method]={'clean':values[0],'noise01':float(np.mean(values[1:])),'C':C,'features':fs}
        rows.append({'train_indices':tr.tolist(),'test_indices':te.tolist(),'choices':choices,'scores':scores})
        print(name,fold,'complete',flush=True)
    assert fingerprints(name)==hashes
    out.write_text(json.dumps({'dataset':name,'d':p['d'],'splits':rows,'fingerprints':hashes},indent=2),encoding='utf-8')


def utility(v):return (v['clean']+v['noise01'])/2


def summarize():
    rows=[]
    for name in SOURCES:
        p=json.loads((FOLDER/f'{name}.json').read_text());s=p['splits'];
        means={m:float(np.mean([utility(x['scores'][m]) for x in s])) for m in ('proposed','full','fgs2023','sfrip2025')}
        rows.append({'dataset':name,'means':means,'removed':1-float(np.mean([len(x['scores']['proposed']['features']) for x in s]))/p['d']})
    comparisons={m:{'gain':float(np.mean([r['means']['proposed']-r['means'][m] for r in rows])),'wins':sum(r['means']['proposed']>r['means'][m]+1e-12 for r in rows)} for m in ('fgs2023','sfrip2025','full')}
    result={'rows':rows,'means':{m:float(np.mean([r['means'][m] for r in rows])) for m in ('proposed','full','fgs2023','sfrip2025')},
            'comparisons':comparisons,'mean_removed':float(np.mean([r['removed'] for r in rows])),
            'gate_passed':all(comparisons[m]['gain']>=.005 and comparisons[m]['wins']>=6 for m in ('fgs2023','sfrip2025')),
            'scope':'eight_previously_exposed_newdata_archived_support_logistic_transfer_not_confirmation'}
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    FOLDER.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,n) for n in SOURCES]):f.result()
    summarize()
