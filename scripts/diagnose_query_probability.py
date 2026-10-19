"""Frozen all-inner eight-table objective comparison, no outer predictions."""
import json
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from diagnose_query_majority import NAMES
from certified_frfs.datasets import load_dataset
from certified_frfs.query_probability import QueryProbabilityObjective,refit_probability_support
from certified_frfs.validated_pool import preprocess_fit,validation_scores


def main():
    threadpool_limits(1);folder=ROOT/'results/query-probability-diagnostic';folder.mkdir(exist_ok=True);out=folder/'all-inner.json';assert not out.exists();rows=[]
    prior=json.loads((ROOT/'results/query-majority-diagnostic/all-inner-fixed-support.json').read_text())
    for name in NAMES:
        p=json.loads((ROOT/'results/predictive-iteration41'/f'{name}-s101-f0.json').read_text());X,y=load_dataset(name);tr=np.array(p['train_indices']);splits=[]
        previous=next(r for r in prior['rows'] if r['dataset']==name)
        for index,s in enumerate(p['inner_splits']):
            ii,jj=np.array(s['train']),np.array(s['validation']);A,B=preprocess_fit(X[tr][ii],X[tr][jj]);objective=QueryProbabilityObjective(X[tr][ii],y[tr][ii]);candidates=[]
            for fraction in [.35,.65,1.]:
                raw=next(c for c in s['fits'] if c['config']['ridge']==.01 and c['config']['fraction']==fraction)
                candidate={k:raw[k] for k in ['config','features','weights','fit']};fitted=refit_probability_support(X[tr][ii],y[tr][ii],candidate,objective=objective)
                scores=validation_scores(A,y[tr][ii],B,y[tr][jj],fitted['features'],np.array(fitted['weights']),144+index,joint=True)
                old=raw['validation'];majority=next(c for c in previous['splits'][index]['candidates'] if c['config']['fraction']==fraction)['metrics']
                utility=lambda r:(r['clean']+r['noise01'])/2
                candidates.append({**fitted,'metrics':scores,'reference_majority_metrics':old,'query_majority_metrics':majority,
                    'delta_reference':utility(scores)-utility(old),'delta_query_majority':utility(scores)-utility(majority)})
            splits.append({'inner':index,'train':ii.tolist(),'validation':jj.tolist(),'candidates':candidates})
        deltas={str(f):float(np.mean([next(c['delta_reference'] for c in s['candidates'] if c['config']['fraction']==f) for s in splits])) for f in [.35,.65,1.]}
        rows.append({'dataset':name,'splits':splits,'deltas':deltas});print(name,deltas,flush=True)
    means={str(f):float(np.mean([r['deltas'][str(f)] for r in rows])) for f in [.35,.65,1.]}
    gates={'each_fraction_mean_gain_at_least_half_point':all(v>=.005 for v in means.values()),'at_least_five_positive_each_fraction':all(sum(r['deltas'][str(f)]>1e-12 for r in rows)>=5 for f in [.35,.65,1.])}
    out.write_text(json.dumps({'rows':rows,'mean_deltas':means,'gates':gates,'gate_passed':all(gates.values()),'scope':'eight_fixed_reused_training_inner_experiments_not_outer_test_superiority'},indent=2),encoding='utf-8');print(json.dumps(means),json.dumps(gates),flush=True)


if __name__=='__main__':main()
