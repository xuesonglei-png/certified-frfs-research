"""Rebuild FGS inner selections and scores from saved inputs."""
import hashlib,json
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from run_softvote_iteration import metric
from run_fgs_nested_comparison import PARAMETERS
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit,validation_scores
from certified_frfs.graph_recent import select_fgs
from review_majority_metric_iteration import score


def main():
    threadpool_limits(1);directory=ROOT/'results/fgs-nested-comparison';inner=outer=0
    for path in sorted(directory.glob('*-s101-f*.json')):
        obj=json.loads(path.read_text());prior=ROOT/'results/predictive-iteration31'/path.name
        assert obj['source_sha256']==hashlib.sha256(prior.read_bytes()).hexdigest()
        p=json.loads(prior.read_text());X,y=load_dataset(obj['dataset']);tr,te=np.array(obj['train_indices']),np.array(obj['test_indices'])
        for c in obj['trace']:
            vals=[]
            for r in c['folds']:
                record=p['inner_splits'][r['inner']];ii,jj=np.array(record['train']),np.array(record['validation']);A,B=preprocess_fit(X[tr][ii],X[tr][jj]);w=np.array(record['weights'][c['family']])
                selected=select_fgs(A,y[tr][ii],**c['config']);assert selected['features']==r['features']
                m=score(A,y[tr][ii],B,y[tr][jj],r['features'],w,144+r['inner']);assert m==r['validation'];vals.append((m['clean']+m['noise01'])/2);inner+=1
            assert abs(np.mean(vals)-c['score'])<1e-12
        chosen=max(obj['trace'],key=lambda c:(c['score'],-c['mean_size'],c['family']=='uniform'));assert chosen==obj['chosen']
        A,B=preprocess_fit(X[tr],X[te]);w=metric(A,y[tr],chosen['family']);assert obj['features']==select_fgs(A,y[tr],**chosen['config'])['features'];assert score(A,y[tr],B,y[te],obj['features'],w,104)==obj['metrics'];outer+=1
    result={'verified_inner_scores':inner,'verified_outer_scores':outer,'status':'independent_FGS_reconstruction_development_only'}
    (directory/'verified-analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))

if __name__=='__main__':main()
