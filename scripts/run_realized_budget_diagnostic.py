"""Exact attribute-count controls using explicit recent budget adaptations."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib,json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_common_fold_recent import ROOT,GRIDS,CS,PATHS,train,score,choose,final_metrics
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from review_box_iterations import paired

FOLDER=ROOT/'results/realized-budget-diagnostic'

def run(name):
    threadpool_limits(1);out=FOLDER/f'{name}.json'
    if out.exists():raise FileExistsError(out)
    paths=[ROOT/'results'/f/f'{name}.json' for f in ('common-fold-recent','nested-teacher-regularized','hierarchical-teacher-selection')]
    common,stages,proposed=[json.loads(p.read_text()) for p in paths];X,y=load_dataset(name);rows=[]
    for i,(s,ps,ss) in enumerate(zip(common['splits'],proposed['splits'],stages['splits'])):
        ri,vi=np.array(s['reference_indices']),np.array(s['validation_indices']);raw,labels=X[ri],y[ri]
        assert ri.tolist()==ps['reference_indices']==ss['reference_indices'] and vi.tolist()==ps['validation_indices']==ss['validation_indices']
        trace={m:[{'config':cfg,'C':C,'folds':[]} for cfg in GRIDS[m] for C in CS] for m in GRIDS}
        for j,inner in enumerate(s['inner']):
            a,b=np.array(inner['reference_indices']),np.array(inner['validation_indices']);A,B=preprocess_fit(raw[a],raw[b])
            size=len(ss['inner'][j]['by_C'][str(ps['teacher_C'])]['stage']['policies'][ps['chosen']]['candidate']['features'])
            for method,choices in trace.items():
                for k,c in enumerate(choices):
                    fs=inner['paths'][method][k//len(CS)]['ranking'][:size]
                    c['folds'].append({'features':fs,'target_size':size,'clean':score(B,labels[b],train(A,labels[a],fs,c['C']))})
        A,B=preprocess_fit(raw,X[vi]);targetsize=len(s['proposed_features']);results={}
        for method,choices in trace.items():
            selected=choose(choices);c=choices[selected];ranking=PATHS[method](A,labels,**c['config'])
            fs=ranking['ranking'][:targetsize];assert len(fs)==targetsize
            fit,metrics=final_metrics(A,labels,B,y[vi],fs,c['C'],144+i)
            results[method]={'chosen_index':selected,'config':c['config'],'C':c['C'],'features':fs,'path':ranking,'fit':fit,'metrics':metrics}
        rows.append({'reference_indices':ri.tolist(),'validation_indices':vi.tolist(),'target_size':targetsize,'trace':trace,'results':results,
            'proposed_LP_metrics':s['proposed_metrics'],'proposed_logistic_metrics':s['results']['proposed_support_logistic_diagnostic']['metrics']})
        print(name,i,'exact attribute budgets scored',flush=True)
    paths += [Path(__file__),ROOT/'research/realized-budget-diagnostic-protocol.md',ROOT/'scripts/run_common_fold_recent.py',ROOT/'src/certified_frfs/recent_budget.py',ROOT/'data/processed'/f'{name}.npz']
    out.write_text(json.dumps({'dataset':name,'d':X.shape[1],'splits':rows,'fingerprints':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}},indent=2),encoding='utf-8')

def summarize(names):
    rows=[]
    for name in names:
        p=json.loads((FOLDER/f'{name}.json').read_text());s=p['splits']
        methods={m:float(np.mean([r['results'][m]['metrics']['clean'] for r in s])) for m in GRIDS}
        methods.update({'proposed_LP':float(np.mean([r['proposed_LP_metrics']['clean'] for r in s])),
                        'proposed_logistic_diagnostic':float(np.mean([r['proposed_logistic_metrics']['clean'] for r in s]))})
        rows.append({'dataset':name,'clean':methods,'removed':float(np.mean([1-r['target_size']/p['d'] for r in s]))})
    result={'rows':rows,'means':{m:float(np.mean([r['clean'][m] for r in rows])) for m in rows[0]['clean']},
        'comparisons':{p:{m:paired([r['clean'][p]-r['clean'][m] for r in rows]) for m in GRIDS} for p in ('proposed_LP','proposed_logistic_diagnostic')},
        'scope':'exact_cardinality_post_selection_recent_adaptations_development_diagnostic_not_native_or_confirmation'}
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print({k:v for k,v in result.items() if k!='rows'},flush=True)

if __name__=='__main__':
    FOLDER.mkdir(exist_ok=True);names=[r['dataset'] for r in json.loads((ROOT/'results/common-fold-recent/complete.json').read_text())['rows']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,n) for n in names]):f.result()
    summarize(names)
