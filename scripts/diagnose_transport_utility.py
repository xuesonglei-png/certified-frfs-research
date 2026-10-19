"""Fixed-subset training-crossfit comparison, no outer labels/scores."""
import json
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.transport_utility import CrossfitTransportUtility
from certified_frfs.cached_transfer_utility import CachedCrossfitTransferUtility


def main():
    threadpool_limits(1);records=json.loads((ROOT/'results/redistributed-metric-diagnostic/fold0-training.json').read_text())['rows'];rows=[]
    for r in records:
        o=json.loads((ROOT/'results/predictive-iteration36'/f"{r['dataset']}-s101-f0.json").read_text())
        full=next(p for p in o['rows'] if p['method']=='protected');family=full['chosen']['family']
        X,y=load_dataset(r['dataset']);tr=np.array(o['train_indices']);fs=r['features']
        a=CachedCrossfitTransferUtility(X[tr],y[tr],family=family)
        b=CrossfitTransportUtility(X[tr],y[tr],family=family)
        assert a.splits==b.splits;old=a.evaluate(fs);new=b.evaluate(fs)
        row={'dataset':r['dataset'],'family':family,'features':fs,'correlation_score':old,
             'transport_score':new,'utility_delta':new['utility']-old['utility']}
        rows.append(row);print(r['dataset'],row['utility_delta'],flush=True)
    directory=ROOT/'results/metric-transport-diagnostic';out=directory/'training-crossfit.json';assert not out.exists()
    result={'scope':'fixed_subset_adaptively_reused_training_crossfit_not_outer_prediction_advantage',
        'rows':rows,'mean_utility_delta':float(np.mean([r['utility_delta'] for r in rows]))}
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')

if __name__=='__main__':main()
