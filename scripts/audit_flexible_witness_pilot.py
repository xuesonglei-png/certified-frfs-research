"""Pilot certification increment only; no test-label use or prediction tuning."""
import json
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.adaptive_witness import adaptive_bounds
from certified_frfs.flexible_witness import flexible_bounds


def main():
    threadpool_limits(1);rows=[]
    for path in sorted((ROOT/'results/predictive-iteration31').glob('*-s101-f0.json')):
        obj=json.loads(path.read_text())
        if obj['dataset'] not in ('iris','wine','sonar','wdbc','ionosphere'):continue
        X,y=load_dataset(obj['dataset']);tr=np.array(obj['train_indices']);A,_=preprocess_fit(X[tr],X[tr])
        for r in obj['rows']:
            fs=r['features'];w=np.array(r['weights']);old=adaptive_bounds(A,y[tr],fs,w)
            bound=flexible_bounds(A,y[tr],fs,w,pool_size=8)
            assert np.all(bound>=old-1e-12)
            rows.append({'dataset':obj['dataset'],'method':r['method'],
                'refreshed_certified':int((old>1e-10).sum()),'flexible_certified':int((bound>1e-10).sum()),
                'gained':int(((bound>1e-10)&(old<=1e-10)).sum())})
    directory=ROOT/'results/flexible-witness-pilot';directory.mkdir(parents=True,exist_ok=True)
    (directory/'pilot.json').write_text(json.dumps({'status':'training_diagnostic_not_predictive_gain','rows':rows},indent=2),encoding='utf-8')
    print(json.dumps({'models':len(rows),'gained':sum(r['gained'] for r in rows),'rows':rows},indent=2))

if __name__=='__main__':main()
