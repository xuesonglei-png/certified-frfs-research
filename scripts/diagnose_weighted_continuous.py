"""All15 first-inner weighted continuous joint certificate screening."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from diagnose_guarded_support import FOLDER as PRIOR, NAMES
from certified_frfs import baselines
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.joint_milp import joint_certified_metric


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--loss-slack',type=float,default=0.);parser.add_argument('--output-subdir',default='weighted-continuous-diagnostic');args=parser.parse_args()
    if args.output_subdir not in ('weighted-continuous-diagnostic','weighted-continuous-slack-diagnostic'):raise ValueError('unknown diagnostic directory')
    threadpool_limits(1); folder=ROOT/'results'/args.output_subdir;folder.mkdir(exist_ok=True);out=folder/'complete.json'
    if out.exists(): raise FileExistsError(out)
    paths=[Path(__file__),ROOT/'src/certified_frfs/joint_milp.py',ROOT/'research/weighted-continuous-protocol.md']
    if args.loss_slack:paths.append(ROOT/'research/loss-slack-protocol.md')
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}; rows=[]
    for name in NAMES:
        prior=PRIOR/f'{name}.json';p=json.loads(prior.read_text());s=p['splits'][0];X,y=load_dataset(name);ri=np.array(s['reference_indices']);A,_=preprocess_fit(X[ri],X[ri])
        relief=baselines.relieff_scores(A,y[ri],random_state=101);order=np.argsort(-relief,kind='stable');rank=np.empty(len(order),int);rank[order]=np.arange(len(order));costs=1+rank/max(1,len(order)-1)
        begin=time.perf_counter();fit=joint_certified_metric(A,y[ri],continuous=True,support_costs=costs,time_limit=10.,loss_slack=args.loss_slack)
        row={'dataset':name,'d':A.shape[1],'relief_scores':relief.tolist(),'support_costs':costs.tolist(),'fit':fit,'seconds':time.perf_counter()-begin,'prior_sha256':hashlib.sha256(prior.read_bytes()).hexdigest(),'data_sha256':hashlib.sha256((ROOT/'data/processed'/f'{name}.npz').read_bytes()).hexdigest()};rows.append(row)
        print(name,'features',len(fit['features']),'/',len(A[0]),'fallback',fit['full_fallback'],'status',fit['fit']['solver_status'],flush=True)
    removed=float(np.mean([1-len(r['fit']['features'])/r['d'] for r in rows]));nonfull=sum(len(r['fit']['features'])<r['d'] for r in rows);gates={'five_nonfull_datasets':nonfull>=5,'mean_removed_at_least_fifteen_percent':removed>=.15}
    assert hashes=={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    result={'complete':True,'rows':rows,'mean_removed':removed,'nonfull_datasets':nonfull,'gates':gates,'gate_passed':all(gates.values()),'source_hashes':hashes,'scope':'weighted_continuous_training_only_joint_certificate_not_prediction_advantage'};out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)

if __name__=='__main__':main()
