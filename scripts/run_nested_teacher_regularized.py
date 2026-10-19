"""Full teacher regularization plus certified support scale selection."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib,json,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import run_nested_teacher_support as base
from run_decision_iteration import ROOT
from certified_frfs.datasets import load_dataset
from certified_frfs.nested_teacher_regularized import fit_stage,inner_splits,FRACTIONS

FOLDER=ROOT/'results/nested-teacher-regularized';CS=(.1,1.,10.)

def select(inner):
    options=[]
    for C in CS:
        for f in FRACTIONS:
            key=str(f);scores=[r['by_C'][str(C)]['policies'][key]['clean'] for r in inner]
            retained=[r['by_C'][str(C)]['policies'][key]['retained'] for r in inner]
            options.append({'C':C,'fraction':key,'score':float(np.mean(scores)),'retained':float(np.mean(retained))})
    best=max(r['score'] for r in options)
    eligible=[r for r in options if r['score']>=best-.005-1e-12]
    proposed=min(eligible,key=lambda r:(r['retained'],-r['score'],r['C'],float(r['fraction'])))
    full=min((r for r in options if r['fraction']=='1.0'),key=lambda r:(-r['score'],r['C']))
    return proposed,full,{'options':options,'eligible':eligible,'tolerance':.005}

def run(name):
    threadpool_limits(1);out=FOLDER/f'{name}.json'
    if out.exists():raise FileExistsError(out)
    priorpath=ROOT/'results/linear-retention'/f'{name}.json';p=json.loads(priorpath.read_text());X,y=load_dataset(name);records=[]
    for i,s in enumerate(p['splits']):
        start=time.perf_counter();ri,vi=np.array(s['reference_indices']),np.array(s['validation_indices']);raw,labels=X[ri],y[ri];inner=[]
        for j,(a,b) in enumerate(inner_splits(labels)):
            fitted={}
            for C in CS:
                stage=fit_stage(raw[a],labels[a],C=C)
                fitted[str(C)]={'stage':stage,'policies':base.score(raw[a],labels[a],raw[b],labels[b],stage)}
            inner.append({'reference_indices':a.tolist(),'validation_indices':b.tolist(),'by_C':fitted})
            print(name,i,'inner',j,'all3C complete',flush=True)
        proposed,full,selection=select(inner);stages={}
        for C in sorted({proposed['C'],full['C']}):stages[str(C)]=fit_stage(raw,labels,C=C)
        selectedstage=stages[str(proposed['C'])];controlstage=stages[str(full['C'])]
        values=base.score(raw,labels,X[vi],y[vi],selectedstage,noise=True,seed=144+i)
        controls=base.score(raw,labels,X[vi],y[vi],controlstage,noise=True,seed=144+i)
        records.append({'reference_indices':ri.tolist(),'validation_indices':vi.tolist(),'inner':inner,'selected_config':proposed,'full_config':full,
            'selection':selection,'outer_stages':stages,'chosen':proposed['fraction'],'stage':selectedstage,
            'metrics':values[proposed['fraction']],'full_metrics':controls['1.0'],
            'same_C_full_metrics':values['1.0'],'seconds':time.perf_counter()-start})
        print(name,i,'chosen',proposed['C'],proposed['fraction'],'BA',values[proposed['fraction']]['clean'],'retained',values[proposed['fraction']]['retained'],flush=True)
    files=[Path(__file__),ROOT/'scripts/run_nested_teacher_support.py',ROOT/'research/nested-teacher-regularized-protocol.md',priorpath,ROOT/'data/processed'/f'{name}.npz']
    files += [ROOT/'src/certified_frfs'/f for f in ('nested_teacher_regularized.py','teacher_support.py','distilled_box_linear_refit.py','linear_retention.py','validated_pool.py','feasible_transport.py','redistributed_loss.py','majority_metric.py','capped_sparse.py')]
    out.write_text(json.dumps({'dataset':name,'d':X.shape[1],'splits':records,'fingerprints':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}},indent=2),encoding='utf-8')

def summarize(names):
    # Keep prior scorer unchanged; adapt only to the expanded inner trace.
    from review_box_iterations import paired
    historical={r['dataset']:r for r in json.loads((ROOT/'results/nested-classifier-pipelines/complete.json').read_text())['rows']};rows=[]
    for name in names:
        p=json.loads((FOLDER/f'{name}.json').read_text());s=p['splits']
        rows.append({'dataset':name,'clean':float(np.mean([r['metrics']['clean'] for r in s])),
            'full':float(np.mean([r['full_metrics']['clean'] for r in s])),'same_C_full':float(np.mean([r['same_C_full_metrics']['clean'] for r in s])),
            'noise':float(np.mean([r['metrics']['query_noise01'] for r in s])),'full_noise':float(np.mean([r['full_metrics']['query_noise01'] for r in s])),
            'removed':float(np.mean([1-r['metrics']['retained'] for r in s])),
            'selected_fallbacks':sum(r['stage']['policies'][r['chosen']]['fallback'] for r in s),
            'inner_fallbacks':sum(t['fallback'] for r in s for inner in r['inner'] for c in inner['by_C'].values() for t in c['stage']['policies'].values()),
            'seconds':sum(r['seconds'] for r in s),
            'historical':{m:historical[name]['means']['logistic'][m]['clean'] for m in ('full','fgs2023','sfrip2025')}})
    comparison=paired([r['clean']-r['full'] for r in rows]);removed=float(np.mean([r['removed'] for r in rows]))
    result={'rows':rows,'mean_clean':float(np.mean([r['clean'] for r in rows])),'mean_full':float(np.mean([r['full'] for r in rows])),
        'mean_removed':removed,'vs_matched_full':comparison,'vs_same_C_full':paired([r['clean']-r['same_C_full'] for r in rows]),
        'vs_matched_full_noise':paired([r['noise']-r['full_noise'] for r in rows]),
        'vs_historical':{m:paired([r['clean']-r['historical'][m] for r in rows]) for m in ('full','fgs2023','sfrip2025')},
        'gate_passed':comparison['mean_pp']>=.5 and comparison['wins']>=10 and removed>=.1,
        'scope':'all15_all44_complete_inner_teacher_regularization_and_support_selection_exposed_development_not_confirmation'}
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print({k:v for k,v in result.items() if k!='rows'},flush=True)

if __name__=='__main__':
    FOLDER.mkdir(exist_ok=True);names=[r['dataset'] for r in json.loads((ROOT/'results/linear-retention/complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,n) for n in names]):f.result()
    summarize(names)
