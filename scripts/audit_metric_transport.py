"""Fixed training-only distance reconstruction and certified feasibility gate."""
import json,hashlib,time
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.metric_transport import fit_transport_weights
from certified_frfs.redistributed_loss import redistributed_loss_bound
from certified_frfs.majority_metric import MajorityTerms
from certified_frfs.core import gamma


def main():
    threadpool_limits(1);source=ROOT/'results/redistributed-metric-diagnostic/fold0-training.json'
    prior=json.loads(source.read_text());rows=[]
    for record in prior['rows']:
        name=record['dataset'];o=json.loads((ROOT/'results/predictive-iteration36'/f'{name}-s101-f0.json').read_text())
        X,y=load_dataset(name);tr=np.array(o['train_indices']);A,_=preprocess_fit(X[tr],X[tr]);fs=record['features'];w=np.array(record['full_weights'])
        t=time.perf_counter();v,info=fit_transport_weights(A,fs,w);dep=redistributed_loss_bound(A,y[tr],fs,w,v)
        protected=MajorityTerms(A,y[tr],weights=w).bounds(range(o['d']),allowed_enemies=2)>1e-10
        bounds=MajorityTerms(A[:,fs],y[tr],weights=v).bounds(range(len(fs)),allowed_enemies=2)
        counts=[int(((bounds>1e-10)&protected&(y[tr]==q)).sum()) for q in np.unique(y[tr])]
        r=next(r for r in o['rows'] if r['method']=='protected');majority=all(c>=need for c,need in zip(counts,r['path']['required']))
        rng=np.random.default_rng(963);wide=np.zeros(o['d']);wide[fs]=v;losses=[]
        for _ in range(20):
            Z=np.clip(A+rng.uniform(-.01,.01,A.shape),0,1)
            losses.append(gamma(Z,y[tr],weights=w)-gamma(Z,y[tr],fs,weights=wide))
        assert max(losses)<=dep['upper_bound']+1e-10
        rows.append({'dataset':name,'features':fs,'weights':v.tolist(),'fit':info,'certificate':dep,
            'joint_certified':majority and dep['upper_bound']<=.1,'retained_counts':counts,
            'sampled_losses':losses,'seconds_including_certificate':time.perf_counter()-t})
        print(name,'mse',info['training_mse_before'],info['training_mse_after'],'joint',rows[-1]['joint_certified'],flush=True)
    before=np.mean([r['fit']['training_mse_before'] for r in rows]);after=np.mean([r['fit']['training_mse_after'] for r in rows])
    gates={'at_least_four_joint_certificates':sum(r['joint_certified'] for r in rows)>=4,
           'objective_never_worse':all(r['fit']['regularized_objective_after']<=r['fit']['regularized_objective_before']+1e-12 for r in rows),
           'mean_mse_relative_improvement_at_least10percent':bool(after<=.9*before)}
    directory=ROOT/'results/metric-transport-diagnostic';directory.mkdir(exist_ok=True);out=directory/'training-gate.json';assert not out.exists()
    result={'scope':'training_only_metric_transport_not_classification_advantage','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'rows':rows,'mean_pair_mse_before':float(before),'mean_pair_mse_after':float(after),
            'gates':gates,'gate_passed':all(gates.values()),'sampled_loss_checks':20*len(rows),'observed_violations':0}
    out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(gates))

if __name__=='__main__':main()
