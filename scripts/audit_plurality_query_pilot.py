"""Fixed label-blind query subsample: plurality vs majority certificates."""
import hashlib,json,time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.plurality_query import certify_plurality_queries
from certified_frfs.knn_query import certify_knn_queries


def main():
    threadpool_limits(1);rows=[]
    for name in ['ecoli','glass','iris','seeds','vehicle','vertebral','wine','yeast']:
        path=ROOT/'results/predictive-iteration26'/f'{name}-s101-f0.json'
        obj=json.loads(path.read_text());X,y=load_dataset(name)
        tr,te=np.array(obj['train_indices']),np.array(obj['test_indices'])
        A,B=preprocess_fit(X[tr],X[te]);ids=np.linspace(0,len(B)-1,min(16,len(B)),dtype=int)
        Q=B[ids];truth=y[te][ids]
        for row in obj['rows']:
            if row['method'] not in ('protected','all'):continue
            fs=row['features'];w=np.array(row['weights'])
            model=KNeighborsClassifier(n_neighbors=5,metric='manhattan',algorithm='brute')
            def predict(ref,query):return model.fit(ref[:,fs]*w[fs],y[tr]).predict(query[:,fs]*w[fs])
            targets=predict(A,Q);start=time.perf_counter()
            plurality=certify_plurality_queries(A,y[tr],Q,targets,fs,weights=w,eps=.01)
            seconds=time.perf_counter()-start
            majority=certify_knn_queries(A,y[tr],Q,targets,fs,weights=w,eps=.01,allowed_enemies=2)
            p=np.array(plurality['certified']);m=np.array(majority['certified'])
            flags=p|m;rng=np.random.default_rng(20261010)
            for _ in range(20):
                ref=np.clip(A+rng.uniform(-.01,.01,A.shape),0,1)
                query=np.clip(Q+rng.uniform(-.01,.01,Q.shape),0,1)
                assert not np.any(flags&(predict(ref,query)!=targets))
            correct=targets==truth
            record={'dataset':name,'method':row['method'],'query_indices':ids.tolist(),
                'nominal_targets':targets.tolist(),'majority':majority,'plurality':plurality,
                'plurality_seconds':seconds,'queries':len(ids),
                'majority_count':int(m.sum()),'plurality_count':int(p.sum()),
                'union_count':int(flags.sum()),'new_plurality_count':int((p&~m).sum()),
                'union_certified_correct':int((flags&correct).sum()),
                'union_certified_wrong':int((flags&~correct).sum()),
                'nominal_correct':int(correct.sum()),'sampled_violations':0,
                'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            rows.append(record)
            print(json.dumps({key:record[key] for key in ['dataset','method','majority_count','plurality_count','new_plurality_count','plurality_seconds']}),flush=True)
    out=ROOT/'results/plurality-query-pilot';out.mkdir(parents=True,exist_ok=True)
    if (out/'pilot.json').exists():raise ValueError('preserve pilot')
    sources=[ROOT/'src/certified_frfs/plurality_query.py',ROOT/'src/certified_frfs/knn_query.py',
             ROOT/'src/certified_frfs/margin.py',ROOT/'scripts/audit_plurality_query_pilot.py']
    (out/'pilot.json').write_text(json.dumps({'rows':rows,
        'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'scope':'posthoc_label_blind_subsample_diagnostic_not_full_test_accuracy_or_superiority'},indent=2),encoding='utf-8')
    print(json.dumps({'models':len(rows),'queries':sum(r['queries'] for r in rows),
        'new_plurality_certificates':sum(r['new_plurality_count'] for r in rows),
        'sampled_violations':0}),flush=True)

if __name__=='__main__':main()
