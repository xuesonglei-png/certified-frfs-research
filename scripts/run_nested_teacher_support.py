"""All15/all44 complete inner teacher/support/certificate/LP refits."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib,json,time
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.nested_teacher_support import FRACTIONS,inner_splits,fit_stage,choose_policy

FOLDER=ROOT/'results/nested-teacher-support'

def score(raw,labels,query,truth,stage,*,noise=False,seed=144):
    _,B=preprocess_fit(raw,query);classes=np.array(stage['teacher_classes']);queries=[B]
    if noise:
        rng=np.random.default_rng(seed);queries += [np.clip(B+rng.uniform(-.01,.01,B.shape),0,1) for _ in range(3)]
    values={}
    for key,r in stage['policies'].items():
        W,b=np.array(r['fit']['coef']),np.array(r['fit']['intercept'])
        ba=[float(balanced_accuracy_score(truth,classes[(Q@W.T+b).argmax(axis=1)])) for Q in queries]
        values[key]={'clean':ba[0],'query_noise01':float(np.mean(ba[1:])) if noise else None,
                     'retained':len(r['candidate']['features'])/stage['d']}
    return values

def run(name):
    threadpool_limits(1);out=FOLDER/f'{name}.json'
    if out.exists():raise FileExistsError(out)
    priorpath=ROOT/'results/linear-retention'/f'{name}.json';prior=json.loads(priorpath.read_text());X,y=load_dataset(name);records=[]
    for i,s in enumerate(prior['splits']):
        start=time.perf_counter();ri,vi=np.array(s['reference_indices']),np.array(s['validation_indices']);raw,labels=X[ri],y[ri];inner=[]
        for j,(a,b) in enumerate(inner_splits(labels)):
            stage=fit_stage(raw[a],labels[a]);values=score(raw[a],labels[a],raw[b],labels[b],stage)
            inner.append({'reference_indices':a.tolist(),'validation_indices':b.tolist(),'stage':stage,'policies':values})
            print(name,i,'inner',j,'complete',flush=True)
        chosen,selection=choose_policy(inner);stage=fit_stage(raw,labels)
        values=score(raw,labels,X[vi],y[vi],stage,noise=True,seed=144+i)
        records.append({'reference_indices':ri.tolist(),'validation_indices':vi.tolist(),'inner':inner,'chosen':chosen,'selection':selection,
                        'stage':stage,'metrics':values[chosen],'full_metrics':values['1.0'],'seconds':time.perf_counter()-start})
        print(name,i,'chosen',chosen,'BA',values[chosen]['clean'],'retained',values[chosen]['retained'],flush=True)
    files=[Path(__file__),ROOT/'research/nested-teacher-support-protocol.md',priorpath,ROOT/'data/processed'/f'{name}.npz']
    files += [ROOT/'src/certified_frfs'/f for f in ('nested_teacher_support.py','teacher_support.py','distilled_box_linear_refit.py','linear_retention.py','validated_pool.py','feasible_transport.py','redistributed_loss.py','majority_metric.py','capped_sparse.py')]
    out.write_text(json.dumps({'dataset':name,'d':X.shape[1],'splits':records,'fingerprints':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}},indent=2),encoding='utf-8')

def summarize(names):
    from review_box_iterations import paired
    historical={r['dataset']:r for r in json.loads((ROOT/'results/nested-classifier-pipelines/complete.json').read_text())['rows']};rows=[]
    for name in names:
        p=json.loads((FOLDER/f'{name}.json').read_text());s=p['splits']
        rows.append({'dataset':name,'clean':float(np.mean([r['metrics']['clean'] for r in s])),
            'full':float(np.mean([r['full_metrics']['clean'] for r in s])),
            'noise':float(np.mean([r['metrics']['query_noise01'] for r in s])),
            'full_noise':float(np.mean([r['full_metrics']['query_noise01'] for r in s])),
            'removed':float(np.mean([1-r['metrics']['retained'] for r in s])),
            'selected_fallbacks':sum(r['stage']['policies'][r['chosen']]['fallback'] for r in s),
            'inner_fallbacks':sum(t['fallback'] for r in s for inner in r['inner'] for t in inner['stage']['policies'].values()),
            'seconds':sum(r['seconds'] for r in s),
            'historical':{m:historical[name]['means']['logistic'][m]['clean'] for m in ('full','fgs2023','sfrip2025')}})
    comparison=paired([r['clean']-r['full'] for r in rows]);removed=float(np.mean([r['removed'] for r in rows]))
    result={'rows':rows,'mean_clean':float(np.mean([r['clean'] for r in rows])),'mean_full':float(np.mean([r['full'] for r in rows])),
        'mean_removed':removed,'vs_matched_full':comparison,'vs_matched_full_noise':paired([r['noise']-r['full_noise'] for r in rows]),
        'vs_historical':{m:paired([r['clean']-r['historical'][m] for r in rows]) for m in ('full','fgs2023','sfrip2025')},
        'gate_passed':comparison['mean_pp']>=.5 and comparison['wins']>=10 and removed>=.1,
        'scope':'all15_all44_complete_inner_refitting_exposed_development_descriptive_not_confirmation'}
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print({k:v for k,v in result.items() if k!='rows'},flush=True)

if __name__=='__main__':
    FOLDER.mkdir(exist_ok=True);names=[r['dataset'] for r in json.loads((ROOT/'results/linear-retention/complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,n) for n in names]):f.result()
    summarize(names)
