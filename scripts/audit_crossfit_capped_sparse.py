"""Crossfit split/score and final fresh certificate checks."""
import json,hashlib,zipfile
import numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from diagnose_crossfit_capped_sparse import ROOT,FOLDER
from audit_trajectory_recent import scaling
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import validation_scores
from certified_frfs.feasible_transport import weight_certificate


def main():
    threadpool_limits(1);counts={'splits':0,'crossfit_scores':0,'fold_certificates':0,'final_certificates':0,'external_scores':0}
    for path in sorted(FOLDER.glob('*.json')):
        if path.name.startswith('complete'):continue
        p=json.loads(path.read_text());name=p['dataset']
        for relative,value in p['fingerprints'].items():assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==value,relative
        prior=json.loads((ROOT/'results/long-capped-sparse'/f'{name}.json').read_text());X,y=load_dataset(name)
        for index,s in enumerate(p['splits']):
            old=prior['splits'][index];ri,vi=np.array(s['reference_indices']),np.array(s['validation_indices'])
            assert ri.tolist()==old['reference_indices'] and vi.tolist()==old['validation_indices'] and set(ri).isdisjoint(vi)
            raw,labels=X[ri],y[ri];fit=s['fit'];classes,n=np.unique(labels,return_counts=True)
            repeated=np.flatnonzero(np.isin(labels,classes[n>1]));single=np.flatnonzero(np.isin(labels,classes[n==1]))
            folds=min(3,int(n[n>1].min()));assert fit['seed']==14100 and fit['folds']==folds
            expected=[(np.r_[repeated[a],single],repeated[b]) for a,b in StratifiedKFold(folds,shuffle=True,random_state=14100).split(repeated,labels[repeated])]
            assert [row['recipe'] for row in fit['summary']]==['full','3','10']
            for row in fit['summary']:
                scores=[];sizes=[]
                for j,record in enumerate(row['folds']):
                    ti,qi=expected[j]
                    assert ti.tolist()==record['train_indices'] and qi.tolist()==record['query_indices'] and set(ti).isdisjoint(qi)
                    A,B=scaling(raw[ti],raw[qi]);fs=record['features'];w=np.array(record['weights'])
                    full_fold=np.ones(p['d'])/np.sqrt(p['d'])
                    assert fs==np.flatnonzero(w>0).tolist()
                    assert w.min()>=0 and w.max()<=2*full_fold.sum()/p['d']+1e-10 and abs(w.sum()-full_fold.sum())<1e-10
                    assert weight_certificate(A,labels[ti],fs,full_fold,w[fs])['feasible']
                    counts['fold_certificates']+=1
                    metrics=validation_scores(A,labels[ti],B,labels[qi],fs,w,14131+j,joint=True)
                    assert all(abs(metrics[k]-record['metrics'][k])<1e-12 for k in metrics)
                    scores.append((metrics['clean']+metrics['noise01'])/2);sizes.append(len(fs));counts['crossfit_scores']+=2
                assert abs(row['utility']-np.mean(scores))<1e-12 and abs(row['mean_size']-np.mean(sizes))<1e-12
            chosen=max(range(3),key=lambda j:(fit['summary'][j]['utility'],-fit['summary'][j]['mean_size'],-j))
            assert fit['chosen_recipe']==fit['summary'][chosen]['recipe']
            A,B=scaling(raw,X[vi]);c=fit['selected'];full=np.ones(p['d'])/np.sqrt(p['d']);w=np.array(c['weights'])
            if fit['chosen_recipe']=='full':
                assert fit['refit'] is None and np.array_equal(w,full)
            else:
                assert fit['selected']==fit['refit']['fit']['compact_selected']
                assert fit['refit']['trajectory']['steps']==int(fit['chosen_recipe'])
            cert=weight_certificate(A,labels,c['features'],full,w[c['features']])
            assert cert['feasible'] and cert==c['certificate'];counts['final_certificates']+=1
            metrics=validation_scores(A,labels,B,y[vi],c['features'],w,144+index,joint=True)
            assert all(abs(metrics[k]-s['metrics'][k])<1e-12 for k in metrics);counts['external_scores']+=2;counts['splits']+=1
        print(name,'crossfit audit complete',flush=True)
    result={**counts,'scope':'split_scaling_choice_score_final_certificate_checks_not_inner_optimizer_refit'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with zipfile.ZipFile(ROOT/'output/crossfit-capped-sparse-v2-evidence.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in FOLDER.glob('*.json'):archive.write(path,str(path.relative_to(ROOT)))
        for path in (ROOT/'src/certified_frfs').glob('*.py'):archive.write(path,str(path.relative_to(ROOT)))
        for rel in ('scripts/diagnose_crossfit_capped_sparse.py','scripts/audit_crossfit_capped_sparse.py','research/crossfit-capped-sparse-protocol.md','tests/test_crossfit_capped_sparse.py'):
            archive.write(ROOT/rel,rel)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
