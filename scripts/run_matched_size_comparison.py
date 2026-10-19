"""Cost-matched MI/Relief/mRMR operating points using training-only budgets.

Budget derives from the proposed training-selected config's inner subsets;
the final budget equals its refitted subset size. Comparators select their
metric family on inner validation. This evaluates conditional operating
points, not an unrestricted baseline's optimal Pareto frontier.
"""
import hashlib,json
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from run_softvote_iteration import metric
from certified_frfs import baselines
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit,validation_scores
from review_majority_metric_iteration import score
from analyze_crossfit_majority_expansion import analyze

METHODS={'mi':baselines.select_mi,'relief':baselines.select_relieff,'mrmr':baselines.select_mrmr}


def main():
    threadpool_limits(1);directory=ROOT/'results/matched-size-comparison';directory.mkdir(parents=True,exist_ok=True)
    grouped={};outer=0
    for source in sorted((ROOT/'results/predictive-iteration31').glob('*-s101-f*.json')):
        out=directory/source.name
        if out.exists():obj=json.loads(out.read_text())
        else:
            prior=json.loads(source.read_text());X,y=load_dataset(prior['dataset']);tr,te=np.array(prior['train_indices']),np.array(prior['test_indices'])
            proposed=next(r for r in prior['rows'] if r['method']=='protected');budgets={f['inner']:len(f['features']) for f in proposed['chosen']['folds']}
            trace={m:[] for m in METHODS}
            for method,function in METHODS.items():
                for family in ('uniform','softvote'):
                    folds=[]
                    for index,record in enumerate(prior['inner_splits']):
                        ii,jj=np.array(record['train']),np.array(record['validation']);A,B=preprocess_fit(X[tr][ii],X[tr][jj]);w=np.array(record['weights'][family])
                        fs=function(A*w,y[tr][ii],k=budgets[index])
                        scores=validation_scores(A,y[tr][ii],B,y[tr][jj],fs,w,144+index,joint=True)
                        assert scores==score(A,y[tr][ii],B,y[tr][jj],fs,w,144+index)
                        folds.append({'inner':index,'features':fs,'validation':scores,'score':(scores['clean']+scores['noise01'])/2})
                    trace[method].append({'family':family,'folds':folds,'score':float(np.mean([f['score'] for f in folds]))})
            A,B=preprocess_fit(X[tr],X[te]);weights={};rows=[];k=len(proposed['features'])
            for method,function in METHODS.items():
                chosen=max(trace[method],key=lambda c:(c['score'],c['family']=='uniform'));family=chosen['family']
                if family not in weights:weights[family]=metric(A,y[tr],family)
                w=weights[family];fs=function(A*w,y[tr],k=k)
                scores=validation_scores(A,y[tr],B,y[te],fs,w,104,joint=True)
                assert scores==score(A,y[tr],B,y[te],fs,w,104) and len(fs)==k
                rows.append({'method':method,'features':fs,'weights':w.tolist(),'chosen':chosen,'trace':trace[method],'metrics':scores})
            rows.append({'method':'protected','features':proposed['features'],'metrics':proposed['metrics']})
            obj={'dataset':prior['dataset'],'fold':prior['fold'],'d':prior['d'],'rows':rows,'budgets':budgets,
                'final_budget':k,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                'status':'conditional_training_selected_equal_size_operating_points_development',
                'independent_score_reconstruction':True}
            out.write_text(json.dumps(obj,indent=2),encoding='utf-8')
            print(prior['dataset'],prior['fold'],'budget',k,[(r['method'],round(r['metrics']['clean'],4)) for r in rows],flush=True)
        assert obj['source_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
        for r in obj['rows']:
            grouped.setdefault(r['method'],{}).setdefault(obj['dataset'],[]).append({**r['metrics'],'removed':1-len(r['features'])/obj['d']});outer+=1
    assert outer==180 and len(grouped['protected'])==15
    result={'status':'development_conditional_matched_size','outer_scores':outer,
        'all15':analyze(grouped,[]),'without_ecoli':analyze(grouped,['ecoli'])}
    (directory/'analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'means':{m:r['means'] for m,r in result['all15']['methods'].items()},'pairs':result['all15']['pairs']},indent=2))

if __name__=='__main__':main()
