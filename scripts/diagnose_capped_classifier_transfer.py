from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit

FOLDER = ROOT/'results/capped-classifier-transfer'
MODELS = ('euclidean5nn', 'rbf_svm', 'logistic')


def model(kind, n):
    if kind == 'euclidean5nn':
        return KNeighborsClassifier(n_neighbors=min(5,n), metric='euclidean', algorithm='brute')
    if kind == 'rbf_svm':
        return SVC(C=1, gamma='scale')
    return LogisticRegression(C=1, max_iter=2000, random_state=0)


def run(name):
    threadpool_limits(1); out = FOLDER/f'{name}.json'
    if out.exists():
        raise FileExistsError(out)
    X, y = load_dataset(name)
    path = ROOT/'results/long-capped-sparse'/f'{name}.json'
    basepath = ROOT/'results/trajectory-recent-comparison'/f'{name}.json'
    prior, base = json.loads(path.read_text()), json.loads(basepath.read_text()); records = []
    for index, (s,b) in enumerate(zip(prior['splits'], base['splits'])):
        assert s['reference_indices'] == b['reference_indices'] and s['validation_indices'] == b['validation_indices']
        ri, vi = np.array(s['reference_indices']), np.array(s['validation_indices'])
        A, B = preprocess_fit(X[ri], X[vi]); d = X.shape[1]
        chosen = s['fit']['compact_selected']; fs = chosen['features']; native = np.array(chosen['weights'])[fs]
        candidates = {'proposed_uniform': (fs, np.ones(len(fs))),
                      'proposed_weighted': (fs, native/native.mean()),
                      'full': (list(range(d)), np.ones(d))}
        candidates.update({m: (b['baselines'][m]['selection']['features'], np.ones(len(b['baselines'][m]['selection']['features'])))
                           for m in ('fgs2023','sfrip2025')})
        rng = np.random.default_rng(144+index); versions = [(A,B)]
        for _ in range(3):
            versions.append((np.clip(A+rng.uniform(-.01,.01,A.shape),0,1), np.clip(B+rng.uniform(-.01,.01,B.shape),0,1)))
        scores = {}
        for kind in MODELS:
            scores[kind] = {}
            for method, (ids,w) in candidates.items():
                values = []
                for ref, query in versions:
                    estimator = model(kind,len(ri)) if ids else DummyClassifier(strategy='prior')
                    R,Q = (ref[:,ids]*w, query[:,ids]*w) if ids else (np.zeros((len(ref),1)),np.zeros((len(query),1)))
                    pred = estimator.fit(R,y[ri]).predict(Q)
                    values.append(float(balanced_accuracy_score(y[vi],pred)))
                scores[kind][method] = {'clean': values[0], 'noise01': float(np.mean(values[1:]))}
        records.append({'reference_indices': ri.tolist(),'validation_indices': vi.tolist(),'scores': scores,
                        'features': {m: ids for m,(ids,_) in candidates.items()}})
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (Path(__file__), ROOT/'research/capped-classifier-transfer-protocol.md',
                         ROOT/'src/certified_frfs/validated_pool.py',path,basepath,ROOT/'data/processed'/f'{name}.npz')}
    out.write_text(json.dumps({'dataset':name,'splits':records,'fingerprints':hashes},indent=2),encoding='utf-8')
    print(name,'complete',flush=True)


def summarize(names):
    rows=[]
    for name in names:
        p=json.loads((FOLDER/f'{name}.json').read_text())
        rows.append({'dataset':name,'means':{kind:{m:{k:float(np.mean([s['scores'][kind][m][k] for s in p['splits']]))
                    for k in ('clean','noise01')} for m in p['splits'][0]['scores'][kind]} for kind in MODELS}})
    result={'rows':rows,'means':{kind:{m:{k:float(np.mean([r['means'][kind][m][k] for r in rows]))
             for k in ('clean','noise01')} for m in rows[0]['means'][kind]} for kind in MODELS},
            'scope':'all15_all44_fixed_support_fixed_classifier_transfer_exposed_development_not_selection_or_test_risk_certification'}
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)


if __name__=='__main__':
    names=[r['dataset'] for r in json.loads((ROOT/'results/long-capped-sparse/complete.json').read_text())['rows']]
    FOLDER.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run,n) for n in names]):
            future.result()
    summarize(names)
