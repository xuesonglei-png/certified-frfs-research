"""Fixed candidate workload incl. subfold fits; detect score/tie drift."""
import json,time
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.redistributed_search import CrossfitRedistributedUtilityWorkspace
from certified_frfs.cached_transfer_utility import CachedCrossfitTransferUtility


def main():
    threadpool_limits(1);rows=[]
    for name in ['iris','wine','sonar','wdbc','ionosphere']:
        o=json.loads((ROOT/'results/predictive-iteration36'/f'{name}-s101-f0.json').read_text())
        X,y=load_dataset(name);tr=np.array(o['train_indices']);X,y=X[tr],y[tr];d=X.shape[1]
        rng=np.random.default_rng(952)
        subsets=[sorted(rng.choice(d,max(1,int(d*f)),replace=False).tolist())
                 for f in [.2,.4,.6,.8] for _ in range(12)]
        elapsed={};scores={}
        for key,constructor in [('uncached',CrossfitRedistributedUtilityWorkspace),('cached',CachedCrossfitTransferUtility)]:
            t=time.perf_counter();ws=constructor(X,y);scores[key]=[ws.evaluate(fs) for fs in subsets]
            elapsed[key]=time.perf_counter()-t
        differences=[{k:scores['cached'][i][k]-scores['uncached'][i][k]
                      for k in ['clean_LOO','shared_noise_LOO','utility']} for i in range(len(subsets))]
        mismatches=sum(any(abs(v)>1e-12 for v in diff.values()) for diff in differences)
        row={'dataset':name,'reference_n':len(X),'d':d,'candidate_calls':len(subsets),
            'elapsed_including_subfold_fits':elapsed,'utility_workload_speed_ratio':elapsed['uncached']/elapsed['cached'],
            'score_mismatch_candidates':mismatches,'max_abs_score_difference':max(abs(v) for diff in differences for v in diff.values()),
            'candidate_differences':differences}
        rows.append(row);print(name,'ratio',row['utility_workload_speed_ratio'],'score mismatches',mismatches,flush=True)
    directory=ROOT/'results/cached-transfer-utility-benchmark';directory.mkdir(exist_ok=True)
    out=directory/'benchmark.json';assert not out.exists(),'preserve evidence'
    out.write_text(json.dumps({'scope':'fixed48_training_candidate_scores_uniform_metric_not_full_search_or_inference',
        'single_timing_run':True,'rows':rows},indent=2),encoding='utf-8')

if __name__=='__main__':main()
