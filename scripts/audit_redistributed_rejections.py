"""Training-only feasibility pilot, frozen one closest rejection per fold0."""
import json,time,hashlib
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.redistributed_loss import transfer_weights,redistributed_loss_bound
from certified_frfs.core import gamma
from certified_frfs.majority_metric import MajorityTerms


def main():
    threadpool_limits(1);rows=[]
    for name in ['iris','wine','sonar','wdbc','ionosphere']:
        source=ROOT/'results/predictive-iteration36'/f'{name}-s101-f0.json'
        o=json.loads(source.read_text());r=next(r for r in o['rows'] if r['method']=='protected')
        candidates=[a for s in r['path']['trace'] for a in s['certification_attempts'] if a['majority_feasible'] and not a['feasible']]
        if not candidates:continue
        c=min(candidates,key=lambda a:(a['dependency_certificate']['upper_bound'],tuple(a['features'])))
        X,y=load_dataset(name);tr=np.array(o['train_indices']);A,_=preprocess_fit(X[tr],X[tr])
        fs=c['features'];wf=np.array(r['weights']);start=time.perf_counter();wr=transfer_weights(A,fs,wf)
        cert=redistributed_loss_bound(A,y[tr],fs,wf,wr);elapsed=time.perf_counter()-start
        wide=np.zeros(o['d']);wide[fs]=wr
        original=MajorityTerms(A,y[tr],weights=wf).bounds(range(o['d']),allowed_enemies=2)>1e-10
        bounds=MajorityTerms(A[:,fs],y[tr],weights=wr).bounds(range(len(fs)),allowed_enemies=2)
        retained=[int(((bounds>1e-10)&original&(y[tr]==label)).sum()) for label in np.unique(y[tr])]
        majority=all(count>=need for count,need in zip(retained,r['path']['required']))
        rng=np.random.default_rng(923);losses=[]
        for _ in range(20):
            Z=np.clip(A+rng.uniform(-.01,.01,A.shape),0,1)
            losses.append(gamma(Z,y[tr],weights=wf)-gamma(Z,y[tr],fs,weights=wide))
        assert max(losses)<=cert['upper_bound']+1e-10
        row={'dataset':name,'fold':0,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
             'features':fs,'full_weights':wf.tolist(),'reduced_weights':wr.tolist(),
             'old_fixed_weight_bound':c['dependency_certificate']['upper_bound'],
             'new_metric_bound':cert['upper_bound'],'majority_retained':retained,'majority_pass':majority,
             'joint_certified':majority and cert['upper_bound']<=.1,'seconds':elapsed,'sampled_losses':losses,
             'certificate':cert}
        rows.append(row);print(name,row['old_fixed_weight_bound'],row['new_metric_bound'],row['joint_certified'],elapsed,flush=True)
    directory=ROOT/'results/redistributed-metric-diagnostic';directory.mkdir(exist_ok=True)
    out=directory/'fold0-training.json'
    if out.exists():raise ValueError('preserve evidence')
    out.write_text(json.dumps({'scope':'changed_metric_training_feasibility_not_prediction_or_old_bound_refinement',
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'src/certified_frfs/redistributed_loss.py',ROOT/'scripts/audit_redistributed_rejections.py']},
        'rows':rows,'sampled_checks':20*len(rows),'sampled_violations':0},indent=2),encoding='utf-8')

if __name__=='__main__':main()
