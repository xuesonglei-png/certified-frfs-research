"""Audit original dependency certificate on new predictive subsets, not repair."""
import json
import numpy as np
from run_decision_iteration import ROOT,scaled_split
from certified_frfs.datasets import load_dataset
from certified_frfs.core import certificate
from threadpoolctl import threadpool_limits


def main():
    threadpool_limits(1)
    directory=ROOT/'results/predictive-iteration15'
    rows=[]
    for path in sorted(directory.glob('*-s*-f*.json')):
        obj=json.loads(path.read_text())
        X,y=load_dataset(obj['dataset'])
        tr=np.array(obj['train_indices'])
        A,_=scaled_split(X[tr],X[tr])
        for row in obj['rows']:
            if row['method'] not in ['graph_robust_majority','graph_nominal_majority','fgs_independent','mrmr']:
                continue
            cert=certificate(A,y[tr],row['features'],eps=.01,method='hybrid')
            rows.append({'dataset':obj['dataset'],'fold':obj['fold'],'method':row['method'],
                         'retained_fraction':len(row['features'])/obj['d'],
                         'original_certificate':cert,'passes_eta01':cert['upper_bound']<=.1+1e-10})
    means={}
    for method in sorted(set(r['method'] for r in rows)):
        members=[r for r in rows if r['method']==method]
        means[method]={'splits':len(members),'passes_eta01':sum(r['passes_eta01'] for r in members),
                       'mean_upper':float(np.mean([r['original_certificate']['upper_bound'] for r in members])),
                       'max_upper':max(r['original_certificate']['upper_bound'] for r in members),
                       'mean_nominal_loss':float(np.mean([r['original_certificate']['nominal_loss'] for r in members]))}
    (directory/'dependency-audit.json').write_text(json.dumps({'rows':rows,'means':means,
        'scope':'original_fixed_weight_shared_box_training_dependency_not_test_accuracy'},indent=2),encoding='utf-8')
    print(json.dumps(means,indent=2))


if __name__=='__main__':
    main()
