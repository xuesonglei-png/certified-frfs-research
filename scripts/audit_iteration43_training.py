"""All saved query training metrics, including rejected final/inner fits."""
import json,hashlib
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from audit_sparse_pilots import preprocessing
from certified_frfs.datasets import load_dataset
from certified_frfs.feasible_transport import weight_certificate
from certified_frfs.core import gamma


def audit(path):
    p=json.loads(path.read_text());X,y=load_dataset(p['dataset']);tr=np.array(p['train_indices']);tables=[];records=[]
    for index,s in enumerate(p['inner_splits']):
        ids=tr[np.array(s['train'])];A,_=preprocessing(X[ids],X[ids]);tables.append((str(index),A,y[ids],s['fits']))
    A,_=preprocessing(X[tr],X[tr]);tables.append(('final',A,y[tr],p['final_candidate_fits']))
    for scope,A,labels,candidates in tables:
        seen=set()
        for c in candidates:
            nodes=[c]+([c['rejected_query_fit']] if 'rejected_query_fit' in c else [])
            for node in nodes:
                fs=node['features'];w=np.array(node['weights']);key=(tuple(fs),tuple(w))
                if key in seen:continue
                seen.add(key);full=np.ones(A.shape[1])/np.sqrt(A.shape[1]);assert abs(w.sum()-full.sum())<1e-10
                cert=weight_certificate(A,labels,fs,full,w[fs]);old=node['certificate']
                assert cert['feasible']==old['feasible'] and cert['retained_counts']==old['retained_counts']
                assert abs(cert['dependency_certificate']['upper_bound']-old['dependency_certificate']['upper_bound'])<1e-12
                if 'trace' in node['fit']:
                    values=[q['objective'] for q in node['fit']['trace']];assert all(b<=a+1e-10 for a,b in zip(values,values[1:]))
                rng=np.random.default_rng(1022);losses=[]
                for _ in range(3):
                    Z=np.clip(A+rng.uniform(-.01,.01,A.shape),0,1);losses.append(gamma(Z,labels,weights=full)-gamma(Z,labels,fs,weights=w))
                assert max(losses)<=cert['dependency_certificate']['upper_bound']+1e-10
                records.append({'scope':scope,'features':fs,'feasible':cert['feasible'],'bound':cert['dependency_certificate']['upper_bound'],'losses':losses})
    return {'result_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'nodes_checked':len(records),'sampled_loss_checks':3*len(records),'observed_violations':0,'rows':records,'scope':'all_saved_training_metrics_certificate/descent/sample_audit_not_independent_optimization'}


def main():
    threadpool_limits(1);folder=ROOT/'results/predictive-iteration43';target=folder/'training-audits';target.mkdir(exist_ok=True)
    for path in sorted(folder.glob('*-s101-f*.json')):
        out=target/path.name
        if out.exists():assert json.loads(out.read_text())['result_sha256']==hashlib.sha256(path.read_bytes()).hexdigest();continue
        result=audit(path);out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(path.name,result['nodes_checked'],'training nodes',flush=True)


if __name__=='__main__':main()
