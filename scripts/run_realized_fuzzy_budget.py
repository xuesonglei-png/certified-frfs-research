"""Exact raw attribute count with identical fuzzy logistic classifier."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib,json
from pathlib import Path
import run_realized_budget_diagnostic as original
from run_common_fuzzy_classifier import train,score,final_metrics

original.FOLDER=original.ROOT/'results/realized-fuzzy-budget'
original.train=train;original.score=score;original.final_metrics=final_metrics

def run(name):
    # Reuse fixed orchestration with a virtual path mapping for one read source.
    # ROOT must remain a real path; the source adaptation is in this copied runner.
    run_adapted(name)

def run_adapted(name):
    import numpy as np
    from threadpoolctl import threadpool_limits
    from certified_frfs.datasets import load_dataset
    from certified_frfs.validated_pool import preprocess_fit
    threadpool_limits(1);out=original.FOLDER/f'{name}.json'
    if out.exists():raise FileExistsError(out)
    paths=[original.ROOT/'results'/f/f'{name}.json' for f in ('common-fuzzy-classifier','nested-teacher-regularized','hierarchical-teacher-selection')]
    common,stages,proposed=[json.loads(p.read_text()) for p in paths];X,y=load_dataset(name);rows=[]
    for i,(s,ps,ss) in enumerate(zip(common['splits'],proposed['splits'],stages['splits'])):
        ri,vi=np.array(s['reference_indices']),np.array(s['validation_indices']);raw,labels=X[ri],y[ri]
        assert ri.tolist()==ps['reference_indices']==ss['reference_indices'] and vi.tolist()==ps['validation_indices']==ss['validation_indices']
        trace={m:[{'config':cfg,'C':C,'folds':[]} for cfg in original.GRIDS[m] for C in original.CS] for m in original.GRIDS}
        for j,inner in enumerate(s['inner']):
            a,b=np.array(inner['reference_indices']),np.array(inner['validation_indices']);A,B=preprocess_fit(raw[a],raw[b])
            size=len(ss['inner'][j]['by_C'][str(ps['teacher_C'])]['stage']['policies'][ps['chosen']]['candidate']['features'])
            for method,choices in trace.items():
                for k,c in enumerate(choices):
                    fs=inner['paths'][method][k//len(original.CS)]['ranking'][:size]
                    c['folds'].append({'features':fs,'target_size':size,'clean':score(B,labels[b],train(A,labels[a],fs,c['C']))})
        A,B=preprocess_fit(raw,X[vi]);targetsize=len(s['proposed_features']);results={}
        for method,choices in trace.items():
            selected=original.choose(choices);c=choices[selected];ranking=original.PATHS[method](A,labels,**c['config'])
            fs=ranking['ranking'][:targetsize];assert len(fs)==targetsize
            fit,metrics=final_metrics(A,labels,B,y[vi],fs,c['C'],144+i)
            results[method]={'chosen_index':selected,'config':c['config'],'C':c['C'],'features':fs,'path':ranking,'fit':fit,'metrics':metrics}
        rows.append({'reference_indices':ri.tolist(),'validation_indices':vi.tolist(),'target_size':targetsize,'trace':trace,'results':results,
            'proposed_LP_metrics':s['proposed_metrics'],'proposed_logistic_metrics':s['results']['proposed_support_logistic_diagnostic']['metrics']})
        print(name,i,'exact fuzzy attribute budget scored',flush=True)
    paths += [Path(__file__),original.ROOT/'research/realized-fuzzy-budget-protocol.md',original.ROOT/'scripts/run_realized_budget_diagnostic.py',original.ROOT/'scripts/run_common_fuzzy_classifier.py',original.ROOT/'src/certified_frfs/triangular_classifier.py',original.ROOT/'data/processed'/f'{name}.npz']
    out.write_text(json.dumps({'dataset':name,'d':X.shape[1],'splits':rows,'fingerprints':{str(p.relative_to(original.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}},indent=2),encoding='utf-8')

if __name__=='__main__':
    original.FOLDER.mkdir(exist_ok=True);names=[r['dataset'] for r in json.loads((original.ROOT/'results/common-fuzzy-classifier/complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,n) for n in names]):f.result()
    original.summarize(names)
