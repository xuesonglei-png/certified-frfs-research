"""Independent score and final-certificate audit for the changed metric."""
import argparse,json,hashlib
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.redistributed_loss import transfer_weights,redistributed_loss_bound
from certified_frfs.majority_metric import MajorityTerms
from run_softvote_iteration import metric
from review_majority_metric_iteration import score
from analyze_crossfit_majority_expansion import analyze


def main(iteration=38,datasets=None):
    threadpool_limits(1);directory=ROOT/f'results/predictive-iteration{iteration}';grouped={};outer=inner=certs=0
    files=sorted(directory.glob('*-s101-f*.json'))
    if datasets is None:
        assert len(files)==5 and {f.name.split('-s101-')[0] for f in files}=={'iris','wine','sonar','wdbc','ionosphere'},'pilot incomplete'
    else:
        files=[f for f in files if f.name.split('-s101-')[0] in datasets]
        assert {f.name.split('-s101-')[0] for f in files}==set(datasets),'selected jobs incomplete'
    for file in files:
        o=json.loads(file.read_text());source=ROOT/f'results/predictive-iteration{38 if iteration==39 else 36}'/file.name
        assert o['baseline_source_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
        old=json.loads(source.read_text());X,y=load_dataset(o['dataset']);tr=np.array(o['train_indices']);te=np.array(o['test_indices']);F,Q=preprocess_fit(X[tr],X[te])
        for r in o['rows']:
            fs=r['features'];w=np.array(r['weights']);assert score(F,y[tr],Q,y[te],fs,w,104)==r['metrics'];outer+=1
            if r['method'] in ('protected','utility'):
                configs=r['trace'];assert len(configs)==12
                for c in configs:
                    values=[];sizes=[]
                    for v in c['folds']:
                        index=v['inner'];rec=o['inner_splits'][index];ii=np.array(rec['train']);jj=np.array(rec['validation'])
                        A,B=preprocess_fit(X[tr][ii],X[tr][jj]);wf=np.array(rec['weights'][c['family']]);wr=transfer_weights(A,v['features'],wf)
                        np.testing.assert_allclose(wr,v['reduced_weights'],rtol=0,atol=1e-12)
                        wide=np.zeros(o['d']);wide[v['features']]=wr
                        metrics=score(A,y[tr][ii],B,y[tr][jj],v['features'],wide,144+index);assert metrics==v['validation']
                        values.append((metrics['clean']+metrics['noise01'])/2);sizes.append(len(v['features']));inner+=1
                    assert abs(np.mean(values)-c['score'])<1e-12 and abs(np.mean(sizes)-c['mean_size'])<1e-12
                assert r['chosen']==max(configs,key=lambda c:(c['score'],-c['mean_size'],c['family']=='uniform'))
                wf=np.array(r['full_weights']);np.testing.assert_allclose(wf,metric(F,y[tr],r['chosen']['family']),rtol=0,atol=1e-12)
                wr=transfer_weights(F,fs,wf);np.testing.assert_allclose(w[fs],wr,rtol=0,atol=1e-12)
                if r['method']=='protected':
                    cert=redistributed_loss_bound(F,y[tr],fs,wf,wr);np.testing.assert_allclose(cert['upper_bound'],r['certificate']['upper_bound'],rtol=0,atol=1e-12);assert cert['upper_bound']<=.1
                    protected=MajorityTerms(F,y[tr],weights=wf).bounds(range(o['d']),allowed_enemies=2)>1e-10
                    bounds=MajorityTerms(F[:,fs],y[tr],weights=wr).bounds(range(len(fs)),allowed_enemies=2)
                    for label,need in zip(np.unique(y[tr]),r['path']['required']):
                        assert ((bounds>1e-10)&protected&(y[tr]==label)).sum()>=need
                    certs+=1
            else:
                prior=next(p for p in old['rows'] if p['method']==r['method'])
                for k in ['features','weights','metrics','chosen','trace']:assert r[k]==prior[k]
            grouped.setdefault(r['method'],{}).setdefault(o['dataset'],[]).append({**r['metrics'],'removed':1-len(fs)/o['d']})
        prior=next(p for p in old['rows'] if p['method']=='protected')
        grouped.setdefault('loo_transfer38' if iteration==39 else 'fixed_dual36',{}).setdefault(o['dataset'],[]).append({**prior['metrics'],'removed':1-len(prior['features'])/o['d']})
        if iteration==39:
            fixed=json.loads((ROOT/'results/predictive-iteration36'/file.name).read_text())
            prior=next(p for p in fixed['rows'] if p['method']=='protected')
            grouped.setdefault('fixed_dual36',{}).setdefault(o['dataset'],[]).append({**prior['metrics'],'removed':1-len(prior['features'])/o['d']})
        print('verified',file.name,flush=True)
    result={'scope':'independent_scores_and_selected_certificates_not_full_path_regeneration_or_confirmation',
        'complete_five_dataset_pilot':len(files)==5,
        'verified_files':[f.name for f in files],
        'verified_outer_scores':outer,'verified_new_inner_scores':inner,'verified_selected_certificates':certs,'analysis':analyze(grouped,[])}
    means={m:r['means'] for m,r in result['analysis']['methods'].items()}
    result['gates']={'clean_at_least_fgs':means['protected']['clean']>=means['fgs']['clean'],
        'clean_gain_over36_at_least_half_point':means['protected']['clean']-means['fixed_dual36']['clean']>=.005,
        'compression_at_least36':means['protected']['removed']>=means['fixed_dual36']['removed']}
    filename='verified-analysis.json' if datasets is None else 'verified-selected-analysis.json'
    (directory/filename).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'counts':[outer,inner,certs],'means':means,'gates':result['gates']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--iteration',type=int,choices=[38,39],default=38)
    p.add_argument('--datasets',nargs='+');a=p.parse_args();main(a.iteration,a.datasets)
