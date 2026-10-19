"""Incremental scoring, selection, fallback and selected certificate audit."""
import json,hashlib
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from audit_sparse_pilots import preprocessing,scoring,equal
from certified_frfs.datasets import load_dataset
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.core import gamma


def audit(path):
    p=json.loads(path.read_text());source=ROOT/'results/predictive-iteration41'/path.name
    assert hashlib.sha256(source.read_bytes()).hexdigest()==p['source41_sha256'];old=json.loads(source.read_text())
    X,y=load_dataset(p['dataset']);tr,te=np.array(p['train_indices']),np.array(p['test_indices'])
    assert p['train_indices']==old['train_indices'] and p['test_indices']==old['test_indices']
    assert set(tr).isdisjoint(te) and set(tr)|set(te)==set(range(len(y)))
    trace={m:{} for m in p['configurations']};inner_count=outer_count=0
    for index,s in enumerate(p['inner_splits']):
        ii,jj=np.array(s['train']),np.array(s['validation']);assert s['train']==old['inner_splits'][index]['train'] and s['validation']==old['inner_splits'][index]['validation']
        A,B=preprocessing(X[tr][ii],X[tr][jj])
        for c in s['fits']:
            scores=scoring(A,y[tr][ii],B,y[tr][jj],c['features'],c['weights'],144+index);equal(scores,c['validation']);inner_count+=1
            if c['method']=='query_certified':assert c['certificate']['feasible']
            if c['full_fallback']:
                np.testing.assert_array_equal(c['weights'],np.ones(A.shape[1])/np.sqrt(A.shape[1]));assert c['features']==list(range(A.shape[1]))
            else:
                for split in c['training_splits']:assert set(split['reference']).isdisjoint(split['query']) and set(split['reference'])|set(split['query'])==set(range(len(ii)))
                assert set(c['features'])<=set(c['allowed_support'])
            trace[c['method']].setdefault(json.dumps(c['config'],sort_keys=True),[]).append({'score':(scores['clean']+scores['noise01'])/2,'size':len(c['features'])})
    for method,configs in p['configurations'].items():
        for c in configs:
            values=trace[method][json.dumps(c['config'],sort_keys=True)]
            assert abs(c['score']-np.mean([r['score'] for r in values]))<1e-12
            assert abs(c['mean_size']-np.mean([r['size'] for r in values]))<1e-12
    F,Q=preprocessing(X[tr],X[te]);checks=[]
    for r in p['rows']:
        c=r['fit'];w=np.array(c['weights']);fs=c['features']
        if r['method'] in p['configurations']:
            chosen=min(p['configurations'][r['method']],key=lambda q:(-q['score'],q['mean_size'],json.dumps(q['config'],sort_keys=True)))
            assert r['chosen']==chosen
            if r['full_fallback']:np.testing.assert_array_equal(w,np.ones(F.shape[1])/np.sqrt(F.shape[1]))
            full=np.ones(F.shape[1])/np.sqrt(F.shape[1]);assert abs(w.sum()-full.sum())<1e-10
            cert=weight_certificate(F,y[tr],fs,full,w[fs]);assert cert['feasible']==c['certificate']['feasible']
            assert abs(cert['dependency_certificate']['upper_bound']-c['certificate']['dependency_certificate']['upper_bound'])<1e-12
            if r['method']=='query_certified':assert cert['feasible']
            rng=np.random.default_rng(1021);losses=[]
            for _ in range(10):
                Z=np.clip(F+rng.uniform(-.01,.01,F.shape),0,1);losses.append(gamma(Z,y[tr],weights=full)-gamma(Z,y[tr],fs,weights=w))
            assert max(losses)<=cert['dependency_certificate']['upper_bound']+1e-10
            checks.append({'method':r['method'],'losses':losses,'bound':cert['dependency_certificate']['upper_bound']})
        equal(scoring(F,y[tr],Q,y[te],fs,w,104),r['metrics']);outer_count+=1
    assert p['baseline_rows']==old['baseline_rows']
    for r in p['baseline_rows']:equal(scoring(F,y[tr],Q,y[te],r['features'],r['weights'],104),r['metrics']);outer_count+=1
    return {'result_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'inner_scores_checked':inner_count,'outer_scores_checked':outer_count,'sampled_loss_checks':20,'observed_violations':0,'checks':checks,'scope':'selected final certificates/score consistency;not optimizer reconstruction/all inner certificate audit'}


def main():
    threadpool_limits(1);folder=ROOT/'results/predictive-iteration43';target=folder/'audits';target.mkdir(exist_ok=True)
    for path in sorted(folder.glob('*-s101-f*.json')):
        out=target/path.name
        if out.exists():assert json.loads(out.read_text())['result_sha256']==hashlib.sha256(path.read_bytes()).hexdigest();continue
        result=audit(path);out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(path.name,'audited',flush=True)


if __name__=='__main__':main()
