"""Independent repeated verification of joint perturbation records."""
import json
import argparse
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.knn_query import certify_knn_queries


def main(iteration=20):
    threadpool_limits(1);directory=ROOT/f'results/joint-perturbation{iteration}'
    grouped={};checks=0;certchecks=0
    for path in sorted(directory.glob('*-s101-f*.json')):
        audit=json.loads(path.read_text());source=json.loads((ROOT/f'results/predictive-iteration{iteration}'/path.name).read_text())
        X,y=load_dataset(source['dataset']);tr=np.array(source['train_indices']);te=np.array(source['test_indices'])
        A,B=preprocess_fit(X[tr],X[te])
        for row in audit['rows']:
            original=next(r for r in source['rows'] if r['method']==row['method'] and r['mode']==row['mode'])
            w=np.array(original['detail']['weights']);fs=row['features']
            def predict(ref,queries):
                return KNeighborsClassifier(n_neighbors=5,metric='manhattan',algorithm='brute').fit(ref[:,fs]*w[fs],y[tr]).predict(queries[:,fs]*w[fs])
            nominal=predict(A,B)
            for record in row['records']:
                eps=record['eps'];cert=certify_knn_queries(A,y[tr],B,nominal,fs,weights=w,eps=eps,k=5)
                assert cert==record['certificate'];certchecks+=1
                flags=np.array(cert['certified']);correct=nominal==y[te]
                def balanced(mask):return float(np.mean([mask[y[te]==c].mean() for c in np.unique(y[te])]))
                assert np.isclose(balanced(flags&correct),record['balanced_certified_correct'])
                assert np.isclose(balanced(flags&~correct),record['balanced_certified_wrong'])
                rng=np.random.default_rng(20261009+source['fold']);values=[];any_wrong=~correct;change=np.zeros(len(B),bool)
                for draw in range(20):
                    ref=np.clip(A+rng.uniform(-eps,eps,A.shape),0,1)
                    query=np.clip(B+rng.uniform(-eps,eps,B.shape),0,1)
                    pred=predict(ref,query);assert not np.any(flags&(pred!=nominal))
                    values.append(float(balanced_accuracy_score(y[te],pred)));any_wrong|=pred!=y[te];change|=pred!=nominal;checks+=1
                assert np.allclose(values,record['joint_draw_balanced_accuracy'],rtol=0,atol=1e-12)
                assert np.isclose(balanced(~any_wrong),record['pointwise_correct_all20_plus_nominal'])
                assert np.isclose(balanced(change),record['balanced_observed_instability'])
                key=f"{row['method']}_{row['mode']}_eps{eps}"
                grouped.setdefault(key,{}).setdefault(source['dataset'],[]).append(record)
    fields=['balanced_certified_correct','balanced_certified_wrong','joint_mean','joint_min_over20','pointwise_correct_all20_plus_nominal','balanced_observed_instability']
    methods={}
    for key,datasets in grouped.items():
        means={name:{field:float(np.mean([r[field] for r in rows])) for field in fields} for name,rows in datasets.items()}
        methods[key]={'datasets':means,'means':{f:float(np.mean([r[f] for r in means.values()])) for f in fields}}
    result={'status':'posthoc_development_not_confirmatory','verified_perturbed_models':checks,'verified_query_certificates':certchecks,'methods':methods}
    (directory/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'verified_models':checks,'verified_certificates':certchecks,'means':{k:v['means'] for k,v in methods.items()}},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--iteration',type=int,choices=[20,21],default=20)
    main(p.parse_args().iteration)
