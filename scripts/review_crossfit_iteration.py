"""Audit reference preprocessing and adaptive-search traces for iteration10."""
import hashlib
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from summarize_predictive_iterations import summarize

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from certified_frfs.datasets import load_dataset

ap=argparse.ArgumentParser()
ap.add_argument('--iteration',type=int,choices=[10,11,12],default=10)
args=ap.parse_args()
directory=ROOT/f'results/predictive-iteration{args.iteration}'
suffix='crossfit' if args.iteration==10 else 'constrained'
compression='accuracy' if args.iteration==10 else 'compression'
summarize(directory)
records=[]
audits=0
shared_pool_checks=0
for path in sorted(directory.glob('*-s*-f*.json')):
    obj=json.loads(path.read_text())
    X,y=load_dataset(obj['dataset'])
    tr=np.array(obj['train_indices'])
    count=min(3,np.unique(y[tr],return_counts=True)[1].min())
    splits=list(StratifiedKFold(n_splits=count,shuffle=True,random_state=obj['seed']+1).split(X[tr],y[tr]))
    tables={}
    for indices in [tr,*[tr[it] for it,iv in splits]]:
        raw,labels=X[indices],y[indices]
        digest=hashlib.sha256(raw.tobytes()+labels.tobytes()).hexdigest()
        tables[digest]=(raw,labels)
    if not obj['stability_records']:
        raise ValueError('missing search traces: this run cannot verify the protocol')
    pools={}
    for search in obj['stability_records']:
        raw,labels=tables[search['training_sha256'][:64]]
        assert search['evaluations']==len(search['trace'])<=24
        assert len({tuple(r['features']) for r in search['trace']})==search['evaluations']
        assert sorted(search['ranking'])==list(range(obj['d']))
        if args.iteration>=11:
            key=search['training_sha256']
            path_signature=[(r['features'],r['objective']) for r in search['trace']]
            if key in pools:
                assert pools[key]==path_signature
                shared_pool_checks+=1
            pools[key]=path_signature
            selection=search['constraint_selection']
            assert selection['selected_predictive_score']>=selection['threshold']-1e-12
            selected=next(row for row in search['trace'] if row['features']==search['features'])
            actual=np.mean([f['balanced_accuracy'] for f in selected['fold_diagnostics']])
            assert abs(actual-selection['selected_predictive_score'])<1e-12
        for fold in search['preprocessing']:
            ref,query=fold['reference_indices'],fold['query_indices']
            assert not set(ref)&set(query)
            assert set(ref)|set(query)==set(range(len(raw)))
            imputer=SimpleImputer(strategy='median',keep_empty_features=True)
            scaled=imputer.fit_transform(raw[ref])
            scaler=MinMaxScaler(clip=True).fit(scaled)
            assert np.allclose(imputer.statistics_,fold['imputer_statistics'],equal_nan=True)
            assert np.allclose(scaler.data_min_,fold['data_min'])
            assert np.allclose(scaler.data_max_,fold['data_max'])
        audits+=1
    rows={r['method']:r for r in obj['rows']}
    record={'dataset':obj['dataset'],'fold':obj['fold'],'d':obj['d'],
            'rows':[{k:r[k] for k in ['method','features','config','metrics','seconds','coverage']}
                    for r in rows.values()]}
    records.append(record)
lines=['# Internal-query search pilot review','','Development evidence; adaptive internal scores are not unbiased estimates.','',
       '| Dataset | Fold | Robust BA | Nominal BA | Pure BA | All BA | Robust removed | Robust seconds |',
       '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
for record in records:
    rows={r['method']:r for r in record['rows']}
    robust=rows['decision_robust_'+suffix]
    ba=lambda method:rows[method]['metrics']['0.0']['balanced_accuracy']
    lines.append(f"| {record['dataset']} | {record['fold']} | {ba('decision_robust_'+suffix):.4f} | "
                 f"{ba('decision_nominal_'+suffix):.4f} | {ba('decision_'+compression+'_'+suffix):.4f} | "
                 f"{ba('all'):.4f} | {1-len(robust['features'])/record['d']:.4f} | {robust['seconds']:.2f} |")
lines+=['',f'Audited {audits} searches: membership partitions, budgets, unique subsets,',
        'reference-only imputer statistics and scaler extrema recomputed from raw source data.',
        'Outer folds select configurations; no test labels generate or score search candidates.',
        'All primary procedures share internal splits, six configs and24-evaluation budgets.',
        'Different objectives may visit different neighborhoods. Classical baselines are cheaper.',
        'Classification superiority and publication readiness remain unproven.',
        'Query stability audit is a separate posthoc measurement, including certified errors.']
lines+=['',f'Shared candidate pool equality checks: {shared_pool_checks}.']
(directory/'pilot-review.json').write_text(json.dumps({'audited_searches':audits,'shared_pool_checks':shared_pool_checks,'records':records},indent=2),encoding='utf-8')
(ROOT/f'research/predictive-iteration{args.iteration}-review.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'completed':len(records),'audited_searches':audits,'shared_pool_checks':shared_pool_checks}))
