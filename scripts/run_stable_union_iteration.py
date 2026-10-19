"""Iteration28: frozen post-selection union transfer, all15 development tasks."""
import hashlib,json
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit,validation_scores
from certified_frfs.stable_union import union_features,certify_union


def main():
    threadpool_limits(1);directory=ROOT/'results/predictive-iteration28';directory.mkdir(parents=True,exist_ok=True)
    for path in sorted((ROOT/'results/predictive-iteration26').glob('*-s101-f*.json')):
        out=directory/path.name
        if out.exists():raise ValueError('preserve iteration28')
        obj=json.loads(path.read_text());X,y=load_dataset(obj['dataset'])
        tr,te=np.array(obj['train_indices']),np.array(obj['test_indices'])
        A,B=preprocess_fit(X[tr],X[te]);rows=[]
        for row in obj['rows']:
            w=np.array(row['weights']);chosen=row['chosen']
            union=union_features([f['features'] for f in chosen['folds']],obj['d'])
            detail=certify_union(A,y[tr],union,w) if row['method']=='protected' else None
            fs=detail['features'] if detail else union
            metrics=validation_scores(A,y[tr],B,y[te],fs,w,104,joint=True)
            rows.append({'method':row['method'],'features':fs,'weights':w.tolist(),
                'union_features':union,'chosen':chosen,'certificate':detail,
                'metrics':metrics,'previous_metrics':row['metrics'],'previous_features':row['features']})
        sources=[ROOT/'src/certified_frfs/stable_union.py',ROOT/'src/certified_frfs/majority_metric.py',
                 ROOT/'scripts/run_stable_union_iteration.py']
        out.write_text(json.dumps({'dataset':obj['dataset'],'fold':obj['fold'],'d':obj['d'],
            'train_indices':tr.tolist(),'test_indices':te.tolist(),'rows':rows,
            'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            'status':'fixed_union_transfer_development_not_direct_inner_validation_of_union'},indent=2),encoding='utf-8')
        print(obj['dataset'],obj['fold'],rows[0]['metrics']['clean'],len(rows[0]['features']),flush=True)

if __name__=='__main__':main()
