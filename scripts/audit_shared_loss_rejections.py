"""Evaluate a valid tighter bound on fixed rejected training subsets."""
import json
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.shared_loss import shared_loss_bound


def main():
    threadpool_limits(1);directory=ROOT/'results/predictive-iteration34'
    data=json.loads((directory/'dependency-rejection-diagnostic.json').read_text());rows=[]
    for file in sorted(directory.glob('*-s101-f*.json')):
        obj=json.loads(file.read_text());X,y=load_dataset(obj['dataset']);tr=np.array(obj['train_indices']);A,_=preprocess_fit(X[tr],X[tr])
        r=next(r for r in obj['rows'] if r['method']=='protected');w=np.array(r['weights'])
        selected=[v for v in data['rows'] if v['dataset']==obj['dataset'] and v['fold']==obj['fold']]
        for previous in selected:
            result=shared_loss_bound(A,y[tr],previous['features'],w)
            assert result['upper_bound']+1e-10>=previous['sampled_max_loss']
            rows.append({**previous,'new_upper_bound':result['upper_bound'],
                'shared_anchor_bound':result['shared_anchor_bound'],
                'newly_certified':result['upper_bound']<=previous['eta']})
        print('shared loss audited',obj['dataset'],obj['fold'],flush=True)
    output={'status':'fixed_training_rejections_no_test_tuning','rows':rows,
        'count':len(rows),'newly_certified':sum(r['newly_certified'] for r in rows),
        'mean_bound_improvement':float(np.mean([r['upper_bound']-r['new_upper_bound'] for r in rows]))}
    (directory/'shared-loss-rejection-audit.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in output.items() if k!='rows'}))

if __name__=='__main__':main()
