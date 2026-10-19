"""Matched-recipe ablation: remove certification, preserve ranks and budgets.

Reuse iteration31 training-fitted metric weights, reconstruct unconstrained
rank/fraction choices on inner training folds, then select without test labels.
"""
import hashlib,json
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from run_softvote_iteration import metric
from certified_frfs import baselines
from certified_frfs.crossfit_majority import FRACTIONS
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit,validation_scores


def candidates(A,y,w):
    d=A.shape[1]
    for rank,relevance in [('mi',baselines.mi_scores(A*w,y,random_state=101)),
                           ('relief',baselines.relieff_scores(A*w,y,random_state=101)),('weights',w)]:
        # Certified backward paths consider ascending relevance for deletion.
        # Reverse exactly this order for the unconstrained matched path.
        order=np.argsort(relevance,kind='stable')[::-1]
        for fraction in FRACTIONS:
            yield {'rank':rank,'fraction':fraction},sorted(order[:max(1,int(np.ceil(fraction*d)))].tolist())


def main():
    threadpool_limits(1);outdir=ROOT/'results/matched-uncertified-ablation';outdir.mkdir(parents=True,exist_ok=True)
    for source in sorted((ROOT/'results/predictive-iteration31').glob('*-s101-f*.json')):
        out=outdir/source.name
        if out.exists():continue
        obj=json.loads(source.read_text());X,y=load_dataset(obj['dataset']);tr=np.array(obj['train_indices']);te=np.array(obj['test_indices'])
        trace={}
        for index,record in enumerate(obj['inner_splits']):
            ii,jj=np.array(record['train']),np.array(record['validation']);A,B=preprocess_fit(X[tr][ii],X[tr][jj])
            for family,weights in record['weights'].items():
                w=np.array(weights);cache={}
                for config,fs in candidates(A,y[tr][ii],w):
                    k=tuple(fs)
                    if k not in cache:cache[k]=validation_scores(A,y[tr][ii],B,y[tr][jj],fs,w,144+index,joint=True)
                    scores=cache[k];key=json.dumps([family,config],sort_keys=True)
                    trace.setdefault(key,{'family':family,'config':config,'folds':[]})['folds'].append({
                        'inner':index,'features':fs,'validation':scores,'score':(scores['clean']+scores['noise01'])/2})
        for c in trace.values():
            c['score']=float(np.mean([r['score'] for r in c['folds']]))
            c['mean_size']=float(np.mean([len(r['features']) for r in c['folds']]))
        chosen=max(trace.values(),key=lambda c:(c['score'],-c['mean_size'],c['family']=='uniform'))
        A,B=preprocess_fit(X[tr],X[te]);w=metric(A,y[tr],chosen['family'])
        fs=next(fs for config,fs in candidates(A,y[tr],w) if config==chosen['config'])
        scores=validation_scores(A,y[tr],B,y[te],fs,w,104,joint=True)
        protected=next(r for r in obj['rows'] if r['method']=='protected')
        out.write_text(json.dumps({'dataset':obj['dataset'],'fold':obj['fold'],'d':obj['d'],
            'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'chosen':chosen,'trace':list(trace.values()),
            'features':fs,'weights':w.tolist(),'metrics':scores,'certified_metrics':protected['metrics'],
            'certified_features':protected['features'],'status':'matched_rank_fraction_metric_budget_development_ablation'},indent=2),encoding='utf-8')
        print(obj['dataset'],obj['fold'],'uncertified',scores['clean'],len(fs),'certified',protected['metrics']['clean'],len(protected['features']),flush=True)

if __name__=='__main__':main()
