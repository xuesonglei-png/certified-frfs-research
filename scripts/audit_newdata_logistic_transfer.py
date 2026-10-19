import hashlib,json
import numpy as np
from threadpoolctl import threadpool_limits
from compare_newdata_logistic_transfer import ROOT,FOLDER,OLD_FOLDER,DATA,SOURCES,CS,score
from certified_frfs.validated_pool import preprocess_fit


def main():
    threadpool_limits(1);counts={'splits':0,'inner_scores':0,'external_scores':0}
    for name in SOURCES:
        p=json.loads((FOLDER/f'{name}.json').read_text());old=json.loads((OLD_FOLDER/f'{name}.json').read_text())
        for rel,h in p['fingerprints'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h
        with np.load(DATA/f'{name}.npz') as z:X,y=z['X'],z['y']
        for fold,s in enumerate(p['splits']):
            saved=old['splits'][fold];tr,te=np.array(s['train_indices']),np.array(s['test_indices'])
            assert tr.tolist()==saved['train_indices'] and te.tolist()==saved['test_indices'] and set(tr).isdisjoint(te)
            raw,labels=X[tr],y[tr];A,B=preprocess_fit(raw,X[te]);rng=np.random.default_rng(144+fold);versions=[(A,B)]
            for _ in range(3):versions.append((np.clip(A+rng.uniform(-.01,.01,A.shape),0,1),np.clip(B+rng.uniform(-.01,.01,B.shape),0,1)))
            for method,result in s['scores'].items():
                fs=result['features']
                expected=saved['fit']['compact_selected']['features'] if method=='proposed' else (list(range(p['d'])) if method=='full' else saved['baselines'][method]['selection']['features'])
                assert fs==expected
                choices=s['choices'][method]
                for c in choices['candidates']:
                    values=[]
                    for inner in saved['fit']['training_splits']:
                        ti,qi=np.array(inner['reference']),np.array(inner['query']);Ai,Bi=preprocess_fit(raw[ti],raw[qi])
                        values.append(score(Ai[:,fs],labels[ti],Bi[:,fs],labels[qi],c['C']))
                    np.testing.assert_allclose(values,c['inner_folds'],atol=1e-12);np.testing.assert_allclose(np.mean(values),c['inner_clean'],atol=1e-12)
                    counts['inner_scores']+=len(values)
                chosen=max(range(len(CS)),key=lambda i:(choices['candidates'][i]['inner_clean'],-len(fs),-i));assert chosen==choices['chosen_index']
                values=[score(R[:,fs],labels,Q[:,fs],y[te],CS[chosen]) for R,Q in versions]
                np.testing.assert_allclose([values[0],np.mean(values[1:])],[result['clean'],result['noise01']],atol=1e-12)
                counts['external_scores']+=4
            counts['splits']+=1
    assert counts['splits']==24
    counts['scope']='archived_support_transfer_replay_not_nested_selector_refits_or_confirmation'
    (FOLDER/'complete-audit.json').write_text(json.dumps(counts,indent=2),encoding='utf-8');print(counts)


if __name__=='__main__':main()
