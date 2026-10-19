"""Training-transfer computational equivalence and timing, not inference speed."""
import json,time,hashlib
import numpy as np
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.cached_transfer import CachedTransfer
from certified_frfs.redistributed_loss import transfer_weights


def main():
    threadpool_limits(1);rows=[]
    for name in ['iris','wine','sonar','wdbc','ionosphere']:
        o=json.loads((ROOT/'results/predictive-iteration36'/f'{name}-s101-f0.json').read_text())
        X,y=load_dataset(name);tr=np.array(o['train_indices']);A,_=preprocess_fit(X[tr],X[tr]);w=np.ones(A.shape[1])/np.sqrt(A.shape[1])
        rng=np.random.default_rng(942);subsets=[sorted(rng.choice(A.shape[1],max(1,int(A.shape[1]*f)),replace=False).tolist())
                for f in [.2,.4,.6,.8] for _ in range(12)]
        t=time.perf_counter();reference=[transfer_weights(A,fs,w) for fs in subsets];old_seconds=time.perf_counter()-t
        t=time.perf_counter();cache=CachedTransfer(A,w);actual=[cache.weights(fs) for fs in subsets];new_seconds=time.perf_counter()-t
        errors=[float(np.max(abs(a-b))) for a,b in zip(reference,actual)]
        assert max(errors)<1e-12
        rows.append({'dataset':name,'n':len(A),'d':A.shape[1],'subsets':len(subsets),
            'uncached_seconds':old_seconds,'cached_including_fit_seconds':new_seconds,
            'transfer_only_speed_ratio':old_seconds/new_seconds,'max_weight_difference':max(errors)})
        print(name,'transfer-only ratio',old_seconds/new_seconds,flush=True)
    directory=ROOT/'results/transfer-cache-benchmark';directory.mkdir(exist_ok=True)
    out=directory/'benchmark.json';assert not out.exists(),'preserve existing timing evidence'
    out.write_text(json.dumps({'scope':'48_training_transfer_calls_not_end_to_end_selection_or_inference_speed',
        'rows':rows,'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [ROOT/'src/certified_frfs/cached_transfer.py',ROOT/'scripts/benchmark_transfer_cache.py']}},indent=2),encoding='utf-8')

if __name__=='__main__':main()
