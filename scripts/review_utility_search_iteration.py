"""Independent prediction, source, nesting and final-retention reconstruction.

Deep mode additionally regenerates new feature paths and metric weights.
Default verification does not claim that all inner searches were rebuilt.
"""
import argparse,hashlib,json
import numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from run_crossfit_majority_iteration import inner_splits
from run_softvote_iteration import metric
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.utility_search import utility_candidates
from certified_frfs.majority_metric import MajorityTerms
from certified_frfs.adaptive_witness import adaptive_bounds
from certified_frfs.core import certificate
from certified_frfs.shared_loss import shared_loss_bound
from certified_frfs.crossfit_utility import CrossfitUtilityWorkspace
from review_majority_metric_iteration import score
from analyze_crossfit_majority_expansion import analyze


def main(deep=False,iteration=33,folds=None):
    threadpool_limits(1);directory=ROOT/f'results/predictive-iteration{iteration}';grouped={};outer=inner=certs=paths=0;verified_files=[]
    for file in sorted(directory.glob('*-s101-f*.json')):
        obj=json.loads(file.read_text());source=ROOT/'results/predictive-iteration31'/file.name
        if folds is not None and obj['fold'] not in folds:continue
        fgs_source=ROOT/'results/fgs-nested-comparison'/file.name
        assert obj['baseline_source_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
        assert obj['fgs_source_sha256']==hashlib.sha256(fgs_source.read_bytes()).hexdigest()
        old=json.loads(source.read_text());X,y=load_dataset(obj['dataset']);tr,te=np.array(obj['train_indices']),np.array(obj['test_indices'])
        a,b=list(StratifiedKFold(3,shuffle=True,random_state=101).split(X,y))[obj['fold']]
        assert np.array_equal(tr,a) and np.array_equal(te,b)
        splits=inner_splits(y[tr]);assert len(splits)==len(obj['inner_splits']);regenerated={};frames={}
        for index,((ii,jj),record) in enumerate(zip(splits,obj['inner_splits'])):
            assert ii.tolist()==record['train'] and jj.tolist()==record['validation']
            assert record['weights']==old['inner_splits'][index]['weights']
            A,B=preprocess_fit(X[tr][ii],X[tr][jj]);frames[index]=(A,B)
            if deep:
                ranking_workspaces={}
                for fit in record['utility_fits']:
                    w=metric(A,y[tr][ii],fit['family']);np.testing.assert_allclose(w,record['weights'][fit['family']],rtol=0,atol=1e-12)
                    w=np.array(record['weights'][fit['family']])
                    if obj.get('crossfit_ranking') and fit['family'] not in ranking_workspaces:
                        ranking_workspaces[fit['family']]=CrossfitUtilityWorkspace(X[tr][ii],y[tr][ii],family=fit['family'])
                    ranking=ranking_workspaces.get(fit['family'])
                    candidates,path=utility_candidates(A,y[tr][ii],w,certified=fit['method']=='protected',dependency_eta=obj.get('dependency_eta') if fit['method']=='protected' else None,shared_loss=obj.get('shared_loss',False) if fit['method']=='protected' else False,ranking_workspace=ranking,beam_width=obj.get('beam_width',1))
                    assert path==fit['path'],(file.name,index,fit['method'],fit['family']);regenerated[index,fit['method'],fit['family']]=candidates;paths+=1
        F,Q=preprocess_fit(X[tr],X[te]);FW={};score_cache={}
        for row in obj['rows']:
            method=row['method']
            if method in ('protected','utility'):
                assert len(row['trace'])==12
                for c in row['trace']:
                    values=[];sizes=[];assert len(c['folds'])==len(splits)
                    for f in c['folds']:
                        index=f['inner'];ii,jj=splits[index];A,B=frames[index]
                        w=np.array(obj['inner_splits'][index]['weights'][c['family']]);fs=f['features']
                        if deep:assert fs==next(v['features'] for v in regenerated[index,method,c['family']] if v['config']==c['config'])
                        k=index,c['family'],tuple(fs)
                        if k not in score_cache:score_cache[k]=score(A,y[tr][ii],B,y[tr][jj],fs,w,144+index)
                        metrics=score_cache[k];assert metrics==f['validation'];values.append((metrics['clean']+metrics['noise01'])/2)
                        assert abs(values[-1]-f['score'])<1e-12;sizes.append(len(fs));inner+=1
                    assert abs(np.mean(values)-c['score'])<1e-12 and abs(np.mean(sizes)-c['mean_size'])<1e-12
                chosen=max(row['trace'],key=lambda c:(c['score'],-c['mean_size'],c['family']=='uniform'));assert chosen==row['chosen']
                family=chosen['family']
                if family not in FW:FW[family]=metric(F,y[tr],family)
                w=FW[family];np.testing.assert_allclose(w,row['weights'],rtol=0,atol=1e-12)
                if deep:
                    ranking=CrossfitUtilityWorkspace(X[tr],y[tr],family=family) if obj.get('crossfit_ranking') else None
                    candidates,path=utility_candidates(F,y[tr],np.array(row['weights']),certified=method=='protected',dependency_eta=obj.get('dependency_eta') if method=='protected' else None,shared_loss=obj.get('shared_loss',False) if method=='protected' else False,ranking_workspace=ranking,beam_width=obj.get('beam_width',1))
                    assert path==row['path'];paths+=1
                    assert row['features']==next(c['features'] for c in candidates if c['config']==chosen['config'])
            else:
                expected=json.loads(fgs_source.read_text()) if method=='fgs' else next(r for r in old['rows'] if r['method']==method)
                for field in ('features','weights','metrics','chosen','trace'):assert row[field]==expected[field]
                w=np.array(row['weights'])
            fs=row['features'];assert score(F,y[tr],Q,y[te],fs,w,104)==row['metrics'];outer+=1
            if method=='protected':
                ws=MajorityTerms(F,y[tr],weights=w);p=ws.bounds(range(obj['d']),allowed_enemies=2)>1e-10
                bounds=np.maximum(ws.bounds(fs,allowed_enemies=2),adaptive_bounds(F,y[tr],fs,w))
                groups=[(y[tr]==label)&p for label in np.unique(y[tr])];c=row['certificate']
                counts=[int(((bounds>1e-10)&g).sum()) for g in groups];required=[int(np.ceil(.9*g.sum()-1e-12)) for g in groups]
                assert c['protected_counts']==[int(g.sum()) for g in groups] and c['required']==required and c['retained_counts']==counts
                assert all(v>=r for v,r in zip(counts,required))
                if obj.get('dependency_eta') is not None:
                    dependency=certificate(F,y[tr],fs,weights=w,eps=.01,method='hybrid')
                    if obj.get('shared_loss',False) and dependency['upper_bound']>obj['dependency_eta']:
                        dependency=shared_loss_bound(F,y[tr],fs,w)
                    assert dependency==c['dependency_certificate'] and dependency['upper_bound']<=obj['dependency_eta']
                np.testing.assert_allclose(np.nan_to_num(bounds,posinf=1e100,neginf=-1e100),c['lower_bounds'],rtol=0,atol=1e-12);certs+=1
            grouped.setdefault(method,{}).setdefault(obj['dataset'],[]).append({**row['metrics'],'removed':1-len(fs)/obj['d']})
        print('verified',obj['dataset'],obj['fold'],flush=True)
        verified_files.append(file.name)
    result={'status':'repeated_development_not_confirmation','deep_path_reconstruction':deep,
        'verified_outer_scores':outer,'verified_new_inner_scores':inner,'verified_training_certificates':certs,
        'reconstructed_paths':paths,'all_datasets':analyze(grouped,[]),'without_ecoli':analyze(grouped,['ecoli'])}
    result['verified_files']=verified_files
    filename='deep-verified-analysis.json' if deep else 'verified-analysis.json'
    if folds is not None:filename=filename.replace('.json','-folds-'+ '-'.join(map(str,folds))+'.json')
    (directory/filename).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'outer':outer,'inner':inner,'certificates':certs,'paths':paths,
        'means':{m:r['means'] for m,r in result['all_datasets']['methods'].items()},'pairs':result['all_datasets']['pairs']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--deep',action='store_true')
    p.add_argument('--iteration',type=int,choices=[33,34,35,36,37],default=33)
    p.add_argument('--folds',type=int,nargs='+');a=p.parse_args();main(a.deep,a.iteration,a.folds)
