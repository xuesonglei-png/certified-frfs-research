"""ALL15 FGS120 native/uniform-count controls; choose only on inner labels."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from expanded_fgs_grid import CONFIGS,LEGACY,expanded_paths
from run_atomic_pruning_classifier import ROOT,digest,final_evaluation
from run_multiscale_certified_metric_knn_pool import evaluate
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from continue_soft_topk_knn_pilot import verify
from audit_forest_compact_transfer import equivalent
from review_box_iterations import paired

SOURCE=ROOT/'results/postmetric-pruning-classifier'
FOLDER=ROOT/'results/expanded-fgs-comparison'
METHODS=('fgs_native120','fgs_uniform_count120')
LEGACY_INDICES=tuple(i for i,c in enumerate(CONFIGS) if c in LEGACY)


def fingerprints(name):
    files=[Path(__file__),ROOT/'scripts/expanded_fgs_grid.py',ROOT/'scripts/audit_expanded_fgs_comparison.py',
        ROOT/'research/expanded-fgs-comparison-protocol.md',ROOT/'research/expanded-fgs-grid-design.md',
        ROOT/'scripts/run_multiscale_certified_metric_knn_pool.py',ROOT/'scripts/run_simplex_soft5_knn_pilot.py',
        ROOT/'scripts/audit_certified_metric_knn_pool.py',SOURCE/(name+'.json'),SOURCE/'complete-audit.json',
        ROOT/'data/processed'/(name+'.npz'),ROOT/'research/literature/pdf/FGS2023Graph.pdf']
    files+=[ROOT/'scripts/audit_atomic_pruning_classifier.py']
    files+=sorted((ROOT/'src/certified_frfs').glob('*.py'))
    return {p.relative_to(ROOT).as_posix():digest(p) for p in files}


def metric(path,method,count):
    fs=path['native']['features'] if method=='fgs_native120' else sorted(path['ranking'][:count])
    return {'features':list(fs),'weights':[1.]*len(fs)}


def scored_fit(A,y,B,truth,count):
    paths=expanded_paths(A,y);records=[];lookup={};options={m:[] for m in METHODS}
    for m in METHODS:
        for row in paths:
            model=metric(row['path'],m,count);key=tuple(model['features'])
            if key not in lookup:
                lookup[key]=len(records);records.append({'metric':model,'evaluation':evaluate(A,y,B,truth,model['features'],model['weights'])})
            options[m].append({'config_index':row['index'],'evaluation_id':lookup[key]})
    return {'paths':paths,'records':records,'options':options,'source_proposed_count':count}


def choose(blocks,m,indices=range(120)):
    rows=[]
    for i in indices:
        selected=[b['records'][b['options'][m][i]['evaluation_id']] for b in blocks]
        rows.append({'config_index':i,'mean_BA':float(np.mean([r['evaluation']['metrics']['clean_BA'] for r in selected])),
            'mean_logloss':float(np.mean([r['evaluation']['metrics']['balanced_logloss'] for r in selected])),
            'mean_count':float(np.mean([len(r['metric']['features']) for r in selected]))})
    best=min(rows,key=lambda r:(-round(r['mean_BA'],12),round(r['mean_logloss'],12),r['mean_count'],r['config_index']))
    return {'rows':rows,'selected':best['config_index']}


def run(name,audit):
    hashes=fingerprints(name);out=FOLDER/(name+'.json')
    if out.exists():
        saved=json.loads(out.read_text());assert saved['fingerprints']==hashes
        if saved['complete']:return saved
    assert digest(SOURCE/(name+'.json'))==audit['artifact_hashes'][name]
    source=json.loads((SOURCE/(name+'.json')).read_text());X,y=load_dataset(name)
    ri,vi=np.array(source['reference_indices']),np.array(source['validation_indices']);assert not set(ri)&set(vi)
    raw,labels=X[ri],y[ri];inners=[]
    for fi,b in enumerate(source['inner']):
        ca,cb=np.array(b['reference_indices']),np.array(b['validation_indices']);A,B=preprocess_fit(raw[ca],raw[cb])
        block=scored_fit(A,labels[ca],B,labels[cb],b['exact_proposed_count'])
        inners.append({'reference_indices':ca.tolist(),'validation_indices':cb.tolist(),**block})
        print(name,'inner',fi+1,'FGS120 complete',len(block['records']),'unique classifier metrics',flush=True)
    selection={m:choose(inners,m) for m in METHODS};legacy={m:choose(inners,m,LEGACY_INDICES) for m in METHODS}
    # Native legacy choice must reproduce the audited existing nine-config
    # model on every inner fold. No outer metric is used to validate selection.
    legacy_native=legacy['fgs_native120']['selected']
    for b,old in zip(inners,source['inner'],strict=True):
        record=b['records'][b['options']['fgs_native120'][legacy_native]['evaluation_id']]
        equivalent(record['evaluation'],old['options']['fgs_native'][0]['evaluation'])
        assert record['metric']['features']==old['options']['fgs_native'][0]['metric']['features']
    A,B=preprocess_fit(raw,X[vi]);paths=expanded_paths(A,labels)
    count=source['outer']['exact_proposed_count'];results={}
    for m in METHODS:
        i=selection[m]['selected'];model=metric(paths[i]['path'],m,count)
        results[m]={'config_index':i,'config':dict(CONFIGS[i]),'metric':model,**final_evaluation(A,labels,B,y[vi],model)}
    assert len(results['fgs_uniform_count120']['metric']['features'])==count
    assert hashes==fingerprints(name)
    saved={'dataset':name,'d':X.shape[1],'reference_indices':ri.tolist(),'validation_indices':vi.tolist(),
        'inner':inners,'selection':selection,'restricted_legacy_selection':legacy,
        'outer':{'paths':paths,'source_proposed_count':count,'results':results},'fingerprints':hashes,
        'complete':True,'query_labels_used_for_selection':False,'goal_achieved':False,
        'scope':'broad_declared_FGS_numerical_native_uniform_count_development_not_author_tables'}
    out.write_text(json.dumps(saved,separators=(',',':')),encoding='utf-8')
    print(name,'FGS120 outer COMPLETE',flush=True);return saved


def summarize(reports):
    rows=[{'dataset':r['dataset'],'d':r['d'],'counts':{m:len(r['outer']['results'][m]['metric']['features']) for m in METHODS},
        'source_proposed_count':r['outer']['source_proposed_count'],
        'metrics':{m:r['outer']['results'][m]['evaluation']['metrics'] for m in METHODS}} for r in reports]
    source=json.loads((SOURCE/'complete.json').read_text());old={r['dataset']:r for r in source['rows']}
    contrasts={m:paired([old[r['dataset']]['metrics']['proposed']['clean_BA']-r['metrics'][m]['clean_BA'] for r in rows]) for m in METHODS}
    legacy_gain=paired([r['metrics']['fgs_native120']['clean_BA']-old[r['dataset']]['metrics']['fgs_native']['clean_BA'] for r in rows])
    return {'rows':rows,'datasets':len(rows),'inner_folds':sum(len(r['inner']) for r in reports),
        'configs_per_fit':120,'means':{m:float(np.mean([r['metrics'][m]['clean_BA'] for r in rows])) for m in METHODS},
        'source_minus_broad_controls':contrasts,'broad_native_minus_legacy_native':legacy_gain,
        'inherited_source_optimizer_failures':source['counts']['inherited_optimizer_failures'],
        'goal_achieved':False,'independent_audit_required':True,
        'limits':['ALL15 reused development; contrasts are descriptive, not confirmatory.',
            'Native/uniform controls do not replace common weighted120-grid controls.',
            'Declared numerical conventions remain distinct from unavailable author artifacts.']}


def main():
    threadpool_limits(1);FOLDER.mkdir(exist_ok=True);audit=verify(SOURCE);names=sorted(audit['artifact_hashes']);assert len(names)==15
    frozen=json.loads((ROOT/'research/expanded-fgs-comparison-source-freeze.json').read_text())
    for rel,sha in frozen['fingerprints'].items():assert digest(ROOT/rel)==sha
    result=summarize([run(n,audit) for n in names]);assert result['inner_folds']==44
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print({k:v for k,v in result.items() if k!='rows'},flush=True)


if __name__=='__main__':main()
