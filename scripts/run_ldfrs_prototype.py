"""Independent fixed-convention LDFRS prototype, not author reproduction."""
import argparse,json,hashlib
import numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT,scaled_split,scores
from certified_frfs.datasets import load_dataset
from certified_frfs.ldfrs import rank_ldfrs


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--convention',choices=['legacy','inclusive_formula','toy_reconciled'],default='legacy')
    args=parser.parse_args()
    threadpool_limits(1);rows=[]
    for name in ['iris','wine','sonar','wdbc','ionosphere']:
        X,y=load_dataset(name)
        tr,te=list(StratifiedKFold(3,shuffle=True,random_state=101).split(X,y))[0]
        splits=list(StratifiedKFold(3,shuffle=True,random_state=102).split(X[tr],y[tr]))
        candidates=[{'fraction':f,'values':[],'features':[]} for f in [.2,.35,.5,.65,.8,1.]]
        for ii,jj in splits:
            A,B=scaled_split(X[tr][ii],X[tr][jj])
            counts=np.unique(y[tr][ii],return_counts=True)[1]
            k=min(10,int(counts.min())-1)
            ranking=rank_ldfrs(A,y[tr][ii],k=k,threshold=1.5,delta=.2,convention=args.convention)
            for candidate in candidates:
                size=max(1,int(np.ceil(candidate['fraction']*X.shape[1])))
                fs=ranking['ranking'][:size]
                metric=scores(A,y[tr][ii],B,y[tr][jj],fs,103,k=5)
                candidate['values'].append((metric['0.0']['balanced_accuracy']+metric['0.01']['balanced_accuracy'])/2)
                candidate['features'].append(fs)
        chosen=max(candidates,key=lambda c:(np.mean(c['values']),-c['fraction']))
        A,B=scaled_split(X[tr],X[te]);k=min(10,int(np.unique(y[tr],return_counts=True)[1].min())-1)
        ranking=rank_ldfrs(A,y[tr],k=k,threshold=1.5,delta=.2,convention=args.convention)
        fs=ranking['ranking'][:max(1,int(np.ceil(chosen['fraction']*X.shape[1])))]
        metric=scores(A,y[tr],B,y[te],fs,104,k=5)
        rows.append({'dataset':name,'fold':0,'seed':101,'d':X.shape[1],
            'train_indices':tr.tolist(),'test_indices':te.tolist(),'features':fs,
            'ranking':ranking,'inner_candidates':candidates,'chosen_fraction':chosen['fraction'],
            'metrics':metric})
        print(json.dumps({'dataset':name,'k':len(fs),'clean':metric['0.0']['balanced_accuracy']}),flush=True)
    suffix='' if args.convention=='legacy' else '-'+args.convention
    directory=ROOT/('results/ldfrs-independent-prototype'+suffix);directory.mkdir(parents=True,exist_ok=True)
    out=directory/'pilot.json'
    if out.exists():raise ValueError('preserve existing prototype')
    out.write_text(json.dumps({'rows':rows,'convention':args.convention,'status':'independent_variant_development_not_official_reproduction',
        'source_hashes':{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [ROOT/'scripts/run_ldfrs_prototype.py',ROOT/'src/certified_frfs/ldfrs.py']}},indent=2),encoding='utf-8')


if __name__=='__main__':main()
