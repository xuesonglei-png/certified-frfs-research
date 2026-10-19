"""Posthoc decomposition of discovery-to-final metric/subset drift."""
import json
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from run_majority_metric_iteration import weights
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit,validation_scores


def main():
    threadpool_limits(1);rows=[]
    for path in sorted((ROOT/'results/predictive-iteration25').glob('*-s101-f*.json')):
        obj=json.loads(path.read_text());X,y=load_dataset(obj['dataset'])
        tr,te=np.array(obj['train_indices']),np.array(obj['test_indices'])
        ii,jj=np.array(obj['discovery_indices']),np.array(obj['validation_indices'])
        A,B=preprocess_fit(X[tr][ii],X[te]);F,Q=preprocess_fit(X[tr],X[te])
        row=next(r for r in obj['rows'] if r['method']=='protected')
        chosen=row['chosen'];w=weights(A,y[tr][ii],chosen['family']);fw=np.array(row['weights'])
        discovery=validation_scores(A,y[tr][ii],B,y[te],chosen['features'],w,104,joint=True)
        refit=validation_scores(F,y[tr],Q,y[te],chosen['features'],fw,104,joint=True)
        final=row['metrics']
        rows.append({'dataset':obj['dataset'],'fold':obj['fold'],'chosen_validation':chosen['validation'],
            'discovery_model':discovery,'refit_before_restoration':refit,'final_model':final,
            'restored':len(row['restoration']),'metric_cosine':float(np.dot(w,fw)),
            'refit_delta':refit['clean']-discovery['clean'],
            'restoration_delta':final['clean']-refit['clean']})
    summary={'models':len(rows),'restored_models':sum(r['restored']>0 for r in rows),
        'refit_delta':float(np.mean([r['refit_delta'] for r in rows])),
        'restoration_delta':float(np.mean([r['restoration_delta'] for r in rows])),
        'metric_cosine':float(np.mean([r['metric_cosine'] for r in rows]))}
    (ROOT/'results/predictive-iteration25/refit-drift.json').write_text(json.dumps({'summary':summary,
        'rows':rows,'scope':'posthoc_development_diagnostic_not_model_selection'},indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
