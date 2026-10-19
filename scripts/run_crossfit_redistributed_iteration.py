"""Iteration39: transferred metric with training-crossfit subset ranking."""
import argparse,copy,hashlib,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit,validation_scores
from certified_frfs.redistributed_search import redistributed_candidates,redistributed_path,CrossfitRedistributedUtilityWorkspace
from certified_frfs.redistributed_loss import transfer_weights,redistributed_loss_bound
from run_softvote_iteration import metric

def wide(d,fs,wr):
    z=np.zeros(d);z[fs]=wr;return z

def run(job):
    threadpool_limits(1);name,fold=job;directory=ROOT/'results/predictive-iteration39';out=directory/f'{name}-s101-f{fold}.json'
    if out.exists():raise ValueError('preserve iteration39')
    source=ROOT/'results/predictive-iteration38'/out.name;old=json.loads(source.read_text());X,y=load_dataset(name);tr=np.array(old['train_indices']);te=np.array(old['test_indices']);d=old['d'];traces={m:{} for m in ['protected','utility']};records=[]
    for index,record in enumerate(old['inner_splits']):
        ii=np.array(record['train']);jj=np.array(record['validation']);A,B=preprocess_fit(X[tr][ii],X[tr][jj]);fits=[]
        for family,values in record['weights'].items():
            wf=np.array(values);ranking=CrossfitRedistributedUtilityWorkspace(X[tr][ii],y[tr][ii],family=family)
            for method in traces:
                start=time.perf_counter();candidates,path=redistributed_candidates(A,y[tr][ii],wf,certified=method=='protected',ranking_workspace=ranking);fits.append({'method':method,'family':family,'path':path,'fit_seconds':time.perf_counter()-start});cache={}
                for c in candidates:
                    fs=c['features'];wr=transfer_weights(A,fs,wf);key=tuple(fs)
                    if key not in cache:cache[key]=validation_scores(A,y[tr][ii],B,y[tr][jj],fs,wide(d,fs,wr),144+index,joint=True)
                    scores=cache[key];k=json.dumps([family,c['config']],sort_keys=True);traces[method].setdefault(k,{'family':family,'config':c['config'],'folds':[]})['folds'].append({'inner':index,'features':fs,'reduced_weights':wr.tolist(),'validation':scores,'score':(scores['clean']+scores['noise01'])/2})
        records.append({**record,'redistributed_fits':fits});print(name,fold,'inner',index,flush=True)
    F,Q=preprocess_fit(X[tr],X[te]);rows=[]
    for method in traces:
        configs=list(traces[method].values());assert len(configs)==12
        for c in configs:c['score']=float(np.mean([v['score'] for v in c['folds']]));c['mean_size']=float(np.mean([len(v['features']) for v in c['folds']]))
        chosen=max(configs,key=lambda c:(c['score'],-c['mean_size'],c['family']=='uniform'));family=chosen['family'];wf=metric(F,y[tr],family);ranking=CrossfitRedistributedUtilityWorkspace(X[tr],y[tr],family=family);candidates,path=redistributed_candidates(F,y[tr],wf,certified=method=='protected',ranking_workspace=ranking);c=next(c for c in candidates if c['config']==chosen['config']);fs=c['features'];wr=transfer_weights(F,fs,wf);weights=wide(d,fs,wr);cert=redistributed_loss_bound(F,y[tr],fs,wf,wr) if method=='protected' else None
        if cert is not None:assert cert['upper_bound']<=.1
        scores=validation_scores(F,y[tr],Q,y[te],fs,weights,104,joint=True);rows.append({'method':method,'features':fs,'weights':weights.tolist(),'full_weights':wf.tolist(),'metrics':scores,'chosen':chosen,'trace':configs,'path':path,'certificate':cert});print(name,fold,method,scores['clean'],len(fs),flush=True)
    rows.extend(copy.deepcopy(r) for r in old['rows'] if r['method'] not in ['protected','utility']);sources=[ROOT/'scripts/run_crossfit_redistributed_iteration.py',ROOT/'src/certified_frfs/redistributed_search.py',ROOT/'src/certified_frfs/redistributed_loss.py'];directory.mkdir(exist_ok=True);out.write_text(json.dumps({'dataset':name,'fold':fold,'d':d,'train_indices':tr.tolist(),'test_indices':te.tolist(),'inner_splits':records,'rows':rows,'baseline_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},'status':'transferred_metric_crossfit_nested_repeated_development_not_confirmation'},indent=2),encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',default=['iris','wine','sonar','wdbc','ionosphere']);p.add_argument('--folds',nargs='+',type=int,default=[0]);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for f in as_completed([pool.submit(run,(n,k)) for n in a.datasets for k in a.folds]):f.result()
