"""Independent sklearn reconstruction of selected deletion ranking scores.

Regenerates subfold preprocessing and weights, checks every accepted move's
ranking score. Does not rebuild all rejected moves or feature searches.
"""
import argparse,json,hashlib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.redistributed_search import CrossfitRedistributedUtilityWorkspace
from certified_frfs.redistributed_loss import transfer_weights


def audit(X,y,family,path):
    ws=CrossfitRedistributedUtilityWorkspace(X,y,family=family)
    assert ws.splits==path['ranking_splits'];count=0;tie_differences=0;score_discrepancies=0
    for step in path['trace']:
        if step['accepted'] is None:continue
        move=next(m for m in step['ranked_moves'] if m['features']==step['accepted'])
        fs=move['features'];scores=[]
        for index,(A,B,yt,yv,w) in enumerate(ws.folds):
            wr=transfer_weights(A,fs,w);rng=np.random.default_rng(144+index)
            Z=np.clip(A+rng.uniform(-.01,.01,A.shape),0,1);Q=np.clip(B+rng.uniform(-.01,.01,B.shape),0,1)
            values=[]
            for ref,query in [(A,B),(Z,Q)]:
                model=KNeighborsClassifier(n_neighbors=min(5,len(yt)),metric='manhattan',algorithm='brute')
                model.fit(ref[:,fs]*wr,yt);sklearn_prediction=model.predict(query[:,fs]*wr)
                D=np.abs(query[:,None,fs]-ref[None,:,fs])@wr
                ids=np.argsort(D,axis=1,kind='stable');k=min(5,len(yt));labels=np.unique(yt)
                votes=np.stack([(yt[ids[:,:k]]==label).sum(axis=1) for label in labels],axis=1)
                stable_prediction=labels[votes.argmax(axis=1)]
                differing=stable_prediction!=sklearn_prediction
                if differing.any():
                    assert k<len(yt),'all-reference votes must agree'
                    gaps=np.take_along_axis(D,ids[:,[k]],axis=1)[:,0]-np.take_along_axis(D,ids[:,[k-1]],axis=1)[:,0]
                    assert np.all(gaps[differing]<=1e-12),'unexplained nontie classifier discrepancy'
                    tie_differences+=int(differing.sum());score_discrepancies+=1
                values.append(balanced_accuracy_score(yv,stable_prediction))
                count+=1
            scores.append(values)
        np.testing.assert_allclose([move['clean_LOO'],move['shared_noise_LOO']],np.mean(scores,axis=0),rtol=0,atol=1e-12)
    return count,tie_differences,score_discrepancies


def main(datasets=None):
    threadpool_limits(1);directory=ROOT/'results/predictive-iteration39';files=sorted(directory.glob('*-s101-f*.json'))
    if datasets is None:assert len(files)==5,'pilot incomplete'
    else:
        files=[f for f in files if f.name.split('-s101-')[0] in datasets]
        assert {f.name.split('-s101-')[0] for f in files}==set(datasets),'requested tasks incomplete'
    rows=[]
    for file in files:
        o=json.loads(file.read_text());X,y=load_dataset(o['dataset']);tr=np.array(o['train_indices']);count=paths=ties=discrepancies=0
        for rec in o['inner_splits']:
            ii=np.array(rec['train'])
            for fit in rec['redistributed_fits']:
                c,t,s=audit(X[tr][ii],y[tr][ii],fit['family'],fit['path']);count+=c;ties+=t;discrepancies+=s;paths+=1
        for r in o['rows']:
            if r['method'] not in ('protected','utility'):continue
            c,t,s=audit(X[tr],y[tr],r['chosen']['family'],r['path']);count+=c;ties+=t;discrepancies+=s;paths+=1
        rows.append({'file':file.name,'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'paths':paths,
            'classifier_scores_checked':count,'near_tie_query_disagreements':ties,'scores_with_sklearn_near_tie_disagreements':discrepancies})
        print('crossfit ranking checked',file.name,flush=True)
    result={'scope':'subfold_fits_and_stable_order_scores_with_sklearn_nontie_agreement_not_full_search_rebuild',
            'complete_five_dataset_pilot':len(files)==5,
            'rows':rows,'totals':{k:sum(r[k] for r in rows) for k in ['paths','classifier_scores_checked','near_tie_query_disagreements','scores_with_sklearn_near_tie_disagreements']}}
    filename='crossfit-ranking-audit.json' if datasets is None else 'crossfit-ranking-audit-selected.json'
    (directory/filename).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result['totals']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+');main(p.parse_args().datasets)
