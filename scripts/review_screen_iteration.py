"""Reconstruct discovery/screen partitions and loss decisions from raw tables."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from certified_frfs.datasets import load_dataset
from certified_frfs.independent_screen import paired_balanced_loss
from summarize_predictive_iterations import summarize
threadpool_limits(1)
directory=ROOT/'results/predictive-iteration13'
summarize(directory)
audited=0; accepted=0; strict=0; nonfull=0
for path in sorted(directory.glob('*-s*-f*.json')):
    obj=json.loads(path.read_text()); X,y=load_dataset(obj['dataset']);tr=np.asarray(obj['train_indices'])
    ni=min(3,np.unique(y[tr],return_counts=True)[1].min())
    splits=list(StratifiedKFold(n_splits=ni,shuffle=True,random_state=obj['seed']+1).split(X[tr],y[tr]))
    tables={}
    for ids in [tr,*[tr[it] for it,iv in splits]]:
        tables[hashlib.sha256(X[ids].tobytes()+y[ids].tobytes()).hexdigest()]=(X[ids],y[ids])
    if not obj['stability_records']:
        raise ValueError('missing screening traces')
    for row in obj['stability_records']:
        raw,labels=tables[row['training_sha256'][:64]]
        a,b=row['discovery_indices'],row['screen_indices']
        assert not set(a)&set(b) and set(a)|set(b)==set(range(len(raw)))
        discovery=row['discovery_search']
        assert discovery['evaluations']<=24
        for fold in discovery['preprocessing']:
            assert all(0<=i<len(a) for i in fold['reference_indices']+fold['query_indices'])
        imp=SimpleImputer(strategy='median',keep_empty_features=True); sc=MinMaxScaler(clip=True)
        A=sc.fit_transform(imp.fit_transform(raw[a])); B=sc.transform(imp.transform(raw[b]))
        def predict(fs):
            return KNeighborsClassifier(n_neighbors=1,metric='manhattan',algorithm='brute').fit(A[:,fs],labels[a]).predict(B[:,fs])
        assessment=paired_balanced_loss(predict(list(range(raw.shape[1]))),predict(row['candidate_features']),labels[b],alpha=.1/6)
        assert assessment==row['assessment']
        assert row['accepted']==(assessment['mean_loss_delta']<=row['tolerance']+1e-12)
        assert row['features']==(row['candidate_features'] if row['accepted'] else list(range(raw.shape[1])))
        audited+=1;accepted+=row['accepted']
        strict+=assessment['upper_loss_delta']<=row['tolerance']+1e-12
        nonfull+=len(row['candidate_features'])<raw.shape[1]
result={'audited':audited,'empirical_acceptance':accepted,'hoeffding_acceptance':strict,
        'nonfull_candidates':nonfull,'scope':'development_empirical_screen_no_final_refit_risk_guarantee'}
(directory/'screen-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
