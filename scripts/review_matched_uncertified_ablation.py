"""Independent scoring and dataset-unit analysis for the matched ablation."""
import hashlib,json
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from run_matched_uncertified_ablation import candidates
from run_softvote_iteration import metric
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from review_majority_metric_iteration import score
from analyze_crossfit_majority_expansion import analyze


def main():
    threadpool_limits(1);directory=ROOT/'results/matched-uncertified-ablation';grouped={};inner=outer=0
    for path in sorted(directory.glob('*-s101-f*.json')):
        source=ROOT/'results/predictive-iteration31'/path.name
        obj=json.loads(path.read_text());assert obj['source_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
        prior=json.loads(source.read_text());X,y=load_dataset(obj['dataset'])
        tr,te=np.array(prior['train_indices']),np.array(prior['test_indices'])
        fitted={};frames={}
        for index,record in enumerate(prior['inner_splits']):
            ii,jj=np.array(record['train']),np.array(record['validation']);A,B=preprocess_fit(X[tr][ii],X[tr][jj])
            frames[index]=(ii,jj,A,B)
            for family,w in record['weights'].items():
                fitted[index,family]={json.dumps(c,sort_keys=True):fs for c,fs in candidates(A,y[tr][ii],np.array(w))}
        for c in obj['trace']:
            scores=[];sizes=[]
            for r in c['folds']:
                index=r['inner'];ii,jj,A,B=frames[index]
                assert r['features']==fitted[index,c['family']][json.dumps(c['config'],sort_keys=True)]
                w=np.array(prior['inner_splits'][index]['weights'][c['family']])
                result=score(A,y[tr][ii],B,y[tr][jj],r['features'],w,144+index)
                assert result==r['validation'];value=(result['clean']+result['noise01'])/2
                assert abs(value-r['score'])<1e-12;scores.append(value);sizes.append(len(r['features']));inner+=1
            assert abs(np.mean(scores)-c['score'])<1e-12 and abs(np.mean(sizes)-c['mean_size'])<1e-12
        chosen=max(obj['trace'],key=lambda c:(c['score'],-c['mean_size'],c['family']=='uniform'))
        assert chosen==obj['chosen'];A,B=preprocess_fit(X[tr],X[te]);w=metric(A,y[tr],chosen['family'])
        np.testing.assert_allclose(w,obj['weights'],rtol=0,atol=1e-12)
        assert obj['features']==next(fs for c,fs in candidates(A,y[tr],w) if c==chosen['config'])
        assert score(A,y[tr],B,y[te],obj['features'],w,104)==obj['metrics'];outer+=1
        p=next(r for r in prior['rows'] if r['method']=='protected')
        assert p['features']==obj['certified_features'] and p['metrics']==obj['certified_metrics']
        for method,metrics,fs in [('protected',p['metrics'],p['features']),('uncertified',obj['metrics'],obj['features'])]:
            grouped.setdefault(method,{}).setdefault(obj['dataset'],[]).append({**metrics,'removed':1-len(fs)/obj['d']})
    result={'verified_inner_scores':inner,'verified_outer_scores':outer,'status':'matched_budget_reused_development',
            'analysis':analyze(grouped,[])}
    (directory/'verified-analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'inner':inner,'outer':outer,'means':{m:r['means'] for m,r in result['analysis']['methods'].items()},
                      'pairs':result['analysis']['pairs']},indent=2))

if __name__=='__main__':main()
