"""Fixed-support crossfit-query metric refit on all15datasets/45tasks.

Support candidates come from immutable iteration41 reference-only fits. Query
weights are refit on disjoint reference/query folds; config selection uses
inner validation only. Certified final candidates are checked against the
    original uniform full metric; failures fall back to the original full
    uniform metric, never to an uncertified learned metric.
"""
import argparse, json, hashlib, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.query_majority import QueryMajorityObjective, refit_query_support
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.validated_pool import preprocess_fit, validation_scores

DATASETS = ['banknote','ecoli','glass','haberman','heart_statlog','ionosphere','iris','seeds','sonar','transfusion','vehicle','vertebral','wdbc','wine','yeast']


def fit_query_candidate(A, y, raw, objective):
    fs = raw['features']; candidate = {k: raw[k] for k in ['config','features','weights','fit']}
    return refit_query_support(A, y, candidate, ridge=.01,objective=objective)


def certify(A, y, fit):
    full = np.ones(A.shape[1])/np.sqrt(A.shape[1]); w = np.asarray(fit['weights'])*full.sum(); fs = fit['features']
    c = weight_certificate(A, y, fs, full, w[fs]); return {**fit, 'weights': w.tolist(), 'full_weights': full.tolist(), 'certificate': c}


def certified_or_full(A, y, fit):
    if fit['certificate']['feasible']: return fit, False
    d=A.shape[1]; full=np.ones(d)/np.sqrt(d); fs=list(range(d))
    cert=weight_certificate(A,y,fs,full,full)
    assert cert['feasible']
    return {'config':fit['config'],'features':fs,'weights':full.tolist(),'full_weights':full.tolist(),
        'certificate':cert,'fit':{'scope':'original_full_uniform_fallback'},
        'rejected_query_fit':fit}, True


def run(job):
    threadpool_limits(1); name, fold = job; folder = ROOT/'results/predictive-iteration43'; folder.mkdir(exist_ok=True)
    out = folder/f'{name}-s101-f{fold}.json'; assert not out.exists()
    source = ROOT/'results/predictive-iteration41'/out.name; old = json.loads(source.read_text()); X, y = load_dataset(name)
    started = time.perf_counter(); tr, te = np.array(old['train_indices']), np.array(old['test_indices']); inner=[]; trace={m:{} for m in ['query_unconstrained','query_certified']}
    for index, split in enumerate(old['inner_splits']):
        ii, jj = np.array(split['train']), np.array(split['validation']); A, B = preprocess_fit(X[tr][ii], X[tr][jj]); fitted=[]
        objective=QueryMajorityObjective(X[tr][ii],y[tr][ii],ridge=.01)
        for raw in split['fits']:
            if raw['config'].get('ridge') != .01: continue
            fit = certify(A, y[tr][ii], fit_query_candidate(X[tr][ii], y[tr][ii], raw,objective))
            protected,fallback=certified_or_full(A,y[tr][ii],fit)
            for method,candidate,fb in [('query_unconstrained',fit,False),('query_certified',protected,fallback)]:
                candidate=dict(candidate)
                scores=validation_scores(A,y[tr][ii],B,y[tr][jj],candidate['features'],np.asarray(candidate['weights']),144+index,joint=True)
                candidate.update({'method':method,'full_fallback':fb,'validation':scores,'score':(scores['clean']+scores['noise01'])/2})
                key=json.dumps(candidate['config'],sort_keys=True);trace[method].setdefault(key,[]).append(candidate);fitted.append(candidate)
        inner.append({'train':ii.tolist(),'validation':jj.tolist(),'fits':fitted})
    configs={m:[{'config':json.loads(k),'score':float(np.mean([r['score'] for r in v])),'mean_size':float(np.mean([len(r['features']) for r in v])),'all_feasible':all(r['certificate']['feasible'] for r in v)} for k,v in records.items()] for m,records in trace.items()}
    ranking=lambda c:(-c['score'],c['mean_size'],json.dumps(c['config'],sort_keys=True))
    F,Q=preprocess_fit(X[tr],X[te]); final=[];objective=QueryMajorityObjective(X[tr],y[tr],ridge=.01)
    for raw in old['final_candidate_fits']:
        if raw['config'].get('ridge') != .01: continue
        final.append(certify(F,y[tr],fit_query_candidate(X[tr],y[tr],raw,objective)))
    rows=[]
    for method in configs:
        chosen=min(configs[method],key=ranking);selected=next(r for r in final if r['config']==chosen['config']);fallback=False
        if method=='query_certified':
            selected,fallback=certified_or_full(F,y[tr],selected);assert selected['certificate']['feasible']
        metrics=validation_scores(F,y[tr],Q,y[te],selected['features'],np.asarray(selected['weights']),104,joint=True)
        rows.append({'method':method,'chosen':chosen,'fit':selected,'metrics':metrics,'full_fallback':fallback})
    raw41=next(r for r in old['rows'] if r['method']=='sparse'); baseline_metrics=raw41['metrics']
    rows.append({'method':'iteration41_raw_control','fit':raw41['fit'],'chosen':raw41['chosen'],'metrics':baseline_metrics})
    sources=sorted((ROOT/'src/certified_frfs').glob('*.py'))+[ROOT/'scripts/run_query_support_iteration43.py']
    payload={'dataset':name,'fold':fold,'d':X.shape[1],'train_indices':tr.tolist(),'test_indices':te.tolist(),'inner_splits':inner,'configurations':configs,'rows':rows,'final_candidate_fits':final,'baseline_rows':old['baseline_rows'],'source41_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'data_sha256':hashlib.sha256((ROOT/'data/processed'/f'{name}.npz').read_bytes()).hexdigest(),'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},'seconds':time.perf_counter()-started,'scope':'development_nested_fixed_support_query_refit_not_untouched_confirmation','certificate_protocol':'original uniform full metric;.1loss/.9retention/.01radius'}
    temporary=out.with_suffix('.json.tmp');temporary.write_text(json.dumps(payload,indent=2),encoding='utf-8');temporary.replace(out);print(name,fold,metrics['clean'],len(selected['features']),selected['certificate']['feasible'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=2);a=p.parse_args();jobs=[(n,f) for n in DATASETS for f in range(3)]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for future in as_completed([pool.submit(run,j) for j in jobs]):future.result()
