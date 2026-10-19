"""Restore empty-producing SFRIP configs with explicit constant prediction."""
import json
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from run_majority_metric_iteration import weights
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit,validation_scores
from certified_frfs.recent import select_sfrip


def evaluate(A,y,B,truth,fs,w,seed):
    if fs:return validation_scores(A,y,B,truth,fs,w,seed,joint=True)
    labels,counts=np.unique(y,return_counts=True)
    pred=np.repeat(labels[counts.argmax()],len(truth))
    value=float(balanced_accuracy_score(truth,pred))
    return {'clean':value,'noise01':value}


def main():
    threadpool_limits(1);directory=ROOT/'results/predictive-iteration26';rows=[]
    for path in sorted(directory.glob('*-s101-f*.json')):
        obj=json.loads(path.read_text());empty=obj.get('empty_inner_outputs',[])
        if not empty:continue
        X,y=load_dataset(obj['dataset']);tr,te=np.array(obj['train_indices']),np.array(obj['test_indices'])
        old=next(r for r in obj['rows'] if r['method']=='sfrip');trace=old['trace'].copy()
        configs={(r['family'],r['config']['beta'],r['config']['confidence']) for r in empty}
        for family,beta,confidence in sorted(configs):
            folds=[]
            for index,split in enumerate(obj['inner_splits']):
                ii,jj=np.array(split['train']),np.array(split['validation'])
                A,B=preprocess_fit(X[tr][ii],X[tr][jj]);w=weights(A,y[tr][ii],family)
                fs=select_sfrip(A*w,y[tr][ii],beta=beta,confidence=confidence)['features']
                metric=evaluate(A,y[tr][ii],B,y[tr][jj],fs,w,144+index)
                folds.append({'inner':index,'features':fs,'validation':metric,
                    'score':(metric['clean']+metric['noise01'])/2})
            trace.append({'family':family,'config':{'beta':beta,'confidence':confidence},'folds':folds,
                'score':float(np.mean([r['score'] for r in folds])),
                'mean_size':float(np.mean([len(r['features']) for r in folds]))})
        chosen=max(trace,key=lambda r:(r['score'],-r['mean_size'],r['family']=='uniform'))
        F,Q=preprocess_fit(X[tr],X[te]);w=weights(F,y[tr],chosen['family'])
        fs=select_sfrip(F*w,y[tr],**chosen['config'])['features']
        metrics=evaluate(F,y[tr],Q,y[te],fs,w,104)
        rows.append({'dataset':obj['dataset'],'fold':obj['fold'],'added_configs':trace[len(old['trace']):],
            'original_chosen':old['chosen'],'constant_endpoint_chosen':chosen,'features':fs,
            'metrics':metrics,'original_metrics':old['metrics'],'selection_changed':chosen!=old['chosen']})
        print(obj['dataset'],obj['fold'],'changed',chosen!=old['chosen'],flush=True)
    (directory/'sfrip-empty-constant-sensitivity.json').write_text(json.dumps({'rows':rows,
        'policy':'empty_subset_predicts_training_majority_class; ties_smallest_label',
        'scope':'development_evaluation_policy_sensitivity_not_published_algorithm_change'},indent=2),encoding='utf-8')

if __name__=='__main__':main()
