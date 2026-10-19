"""Check retained certified LOO decisions under shared sampled perturbations."""
import argparse,json
import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit


def main(iteration=24):
    threadpool_limits(1);directory=ROOT/f'results/predictive-iteration{iteration}';rows=[]
    for path in sorted(directory.glob('*-s101-f*.json')):
        obj=json.loads(path.read_text());X,y=load_dataset(obj['dataset']);tr=np.array(obj['train_indices'])
        A,_=preprocess_fit(X[tr],X[tr]);labels=y[tr]
        row=next(r for r in obj['rows'] if r['method']=='protected')
        fs=row['features'];w=np.array(row['weights']);mask=np.array(row['certificate']['lower_bounds'])>1e-10
        rng=np.random.default_rng(20261009+obj['fold']);violation=0;correct=[]
        for _ in range(20):
            Z=np.clip(A+rng.uniform(-.01,.01,A.shape),0,1)
            D=cdist(Z[:,fs]*w[fs],Z[:,fs]*w[fs],'cityblock');np.fill_diagonal(D,np.inf)
            nearest=np.argsort(D,axis=1,kind='stable')[:,:5]
            pred=np.array([np.unique(labels[js],return_counts=True)[0][np.unique(labels[js],return_counts=True)[1].argmax()] for js in nearest])
            violation+=int(np.sum(mask&(pred!=labels)))
            correct.append(float(np.mean(pred==labels)))
        assert violation==0
        rows.append({'dataset':obj['dataset'],'fold':obj['fold'],'certified':int(mask.sum()),
            'draws':20,'violations':violation,'mean_LOO_accuracy':float(np.mean(correct))})
    (directory/'sampled-LOO-audit.json').write_text(json.dumps({'rows':rows,
        'scope':'sampled_sanity_check_not_exhaustive_box_verification'},indent=2),encoding='utf-8')
    print(json.dumps({'models':len(rows),'draws':sum(r['draws'] for r in rows),
        'certified_object_draws':sum(r['certified']*r['draws'] for r in rows),
        'violations':sum(r['violations'] for r in rows)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--iteration',type=int,choices=[24,25,26,28,29,30,31,32,33],default=24)
    main(p.parse_args().iteration)
