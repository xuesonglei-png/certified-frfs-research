"""Training-only witness-refresh diagnostic, no new test evaluation."""
import json
import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.adaptive_witness import adaptive_bounds


def main():
    threadpool_limits(1);rows=[]
    for path in sorted((ROOT/'results/predictive-iteration26').glob('*-s101-f0.json')):
        obj=json.loads(path.read_text())
        if obj['dataset'] not in ('iris','wine','sonar','wdbc','ionosphere'):continue
        X,y=load_dataset(obj['dataset']);tr=np.array(obj['train_indices']);y=y[tr]
        A,_=preprocess_fit(X[tr],X[tr])
        for row in obj['rows']:
            fs=row['features'];w=np.array(row['weights']);old=np.array(row['certificate']['lower_bounds'])>1e-10
            bounds=adaptive_bounds(A,y,fs,w);new=bounds>1e-10
            violations=0;rng=np.random.default_rng(20261010)
            for _ in range(20):
                Z=np.clip(A+rng.uniform(-.01,.01,A.shape),0,1)
                D=cdist(Z[:,fs]*w[fs],Z[:,fs]*w[fs],'cityblock');np.fill_diagonal(D,np.inf)
                nn=np.argsort(D,axis=1,kind='stable')[:,:5]
                pred=np.array([np.unique(y[ids],return_counts=True)[0][np.unique(y[ids],return_counts=True)[1].argmax()] for ids in nn])
                violations+=int(np.sum(new&(pred!=y)))
            assert violations==0
            rows.append({'dataset':obj['dataset'],'method':row['method'],'features':len(fs),
                'old_certified':int(old.sum()),'new_certified':int(new.sum()),
                'gained':int((new&~old).sum()),'lost':int((old&~new).sum()),'violations':violations})
    out=ROOT/'results/adaptive-witness-pilot';out.mkdir(parents=True,exist_ok=True)
    (out/'pilot.json').write_text(json.dumps({'status':'training_diagnostic_not_predictive_gain','rows':rows},indent=2),encoding='utf-8')
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
