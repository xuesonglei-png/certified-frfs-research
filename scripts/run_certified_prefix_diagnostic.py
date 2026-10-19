"""ALL15/ALL44 inner-only prefix evaluation; no outer score reads."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_simplex_soft5_knn_pilot import ROOT,digest
from continue_soft_topk_knn_pilot import verify
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_path_prefixes import prefixes,PROGRESS
from run_multiscale_certified_metric_knn_pool import evaluate
from audit_forest_compact_transfer import equivalent

SOURCE=ROOT/'results/postmetric-pruning-classifier'
FOLDER=ROOT/'results/certified-prefix-diagnostic'
METHODS=('proposed','unguarded_refined','raw_repair_refined','full_refined','fgs_refined','sfrip_refined')


def fingerprints(name):
    files=[Path(__file__),ROOT/'scripts/certified_path_prefixes.py',ROOT/'scripts/audit_certified_prefix_diagnostic.py',
        ROOT/'research/certified-prefix-diagnostic-protocol.md',ROOT/'scripts/run_multiscale_certified_metric_knn_pool.py',
        ROOT/'scripts/audit_certified_metric_knn_pool.py',SOURCE/(name+'.json'),SOURCE/'complete-audit.json',
        ROOT/'data/processed'/(name+'.npz')]+sorted((ROOT/'src/certified_frfs').glob('*.py'))
    return {p.relative_to(ROOT).as_posix():digest(p) for p in files}


def criterion(r,index):
    return (-round(r['mean_BA'],12),round(r['mean_logloss'],12),r['mean_count'],index)


def recipes(blocks,method,progress):
    rows=[]
    for p in progress:
        chosen=[b['methods'][method]['prefixes'][int(np.ceil(p*(len(b['methods'][method]['prefixes'])-1)))] for b in blocks]
        rows.append({'progress':p,'mean_BA':float(np.mean([r['evaluation']['metrics']['clean_BA'] for r in chosen])),
            'mean_logloss':float(np.mean([r['evaluation']['metrics']['balanced_logloss'] for r in chosen])),
            'mean_count':float(np.mean([len(r['metric']['features']) for r in chosen]))})
    selected=min(range(len(rows)),key=lambda i:criterion(rows[i],i))
    return {'rows':rows,'selected':selected}


def summarize_saved(saved):
    output={}
    for method in METHODS:
        five=recipes(saved['inner'],method,PROGRESS);binary=recipes(saved['inner'],method,(0.,1.))
        fr=five['rows'][five['selected']];br=binary['rows'][binary['selected']]
        fold_gains=[];strict_best=0;intermediates=0
        for b in saved['inner']:
            rows=b['methods'][method]['prefixes'];intermediates+=max(0,len(rows)-2)
            best=max(r['evaluation']['metrics']['clean_BA'] for r in rows)
            boundary=max(rows[0]['evaluation']['metrics']['clean_BA'],rows[-1]['evaluation']['metrics']['clean_BA'])
            fold_gains.append(100*(best-boundary));strict_best+=best>boundary+1e-12
        output[method]={'five':five,'binary':binary,'five_minus_binary_pp':100*(fr['mean_BA']-br['mean_BA']),
            'five_minus_binary_mean_count':fr['mean_count']-br['mean_count'],
            'intermediate_models':intermediates,'folds_with_strictly_better_intermediate':strict_best,
            'optimistic_fold_oracle_mean_extra_pp':float(np.mean(fold_gains))}
    return output


def run(name,source_audit):
    hashes=fingerprints(name);path=FOLDER/(name+'.json')
    if path.exists():
        saved=json.loads(path.read_text());assert saved['fingerprints']==hashes
        if saved['complete']:return saved
    assert digest(SOURCE/(name+'.json'))==source_audit['artifact_hashes'][name]
    source=json.loads((SOURCE/(name+'.json')).read_text())
    X,y=load_dataset(name);ri=np.array(source['reference_indices']);raw,labels=X[ri],y[ri];blocks=[];unique=0
    # Only training-reference indices and inner blocks are passed to analysis.
    for fi,b in enumerate(source['inner']):
        ca,cb=np.array(b['reference_indices']),np.array(b['validation_indices']);assert not set(ca)&set(cb)
        A,B=preprocess_fit(raw[ca],raw[cb]);models={};cache={}
        for method in METHODS:
            parent,terminal=b['options'][method];pid=terminal['metric']['pruning_id'];fit=b['pruning_records'][pid]['fit']
            path_rows=prefixes(fit,require_original=method=='proposed');scores=[]
            for metric in path_rows:
                key=(tuple(metric['features']),tuple(metric['weights']))
                if key not in cache:cache[key]=evaluate(A,labels[ca],B,labels[cb],metric['features'],metric['weights'])
                small={k:metric[k] for k in ('features','weights','path_step','guard_reference')}
                scores.append({'metric':small,'evaluation':cache[key]})
            equivalent(scores[0]['evaluation'],parent['evaluation']);equivalent(scores[-1]['evaluation'],terminal['evaluation'])
            models[method]={'source_pruning_id':pid,'prefixes':scores}
        unique+=len(cache);blocks.append({'reference_indices':ca.tolist(),'validation_indices':cb.tolist(),'methods':models})
        print(name,'inner',fi+1,'all chosen prefixes evaluated',len(cache),'unique metrics',flush=True)
    assert hashes==fingerprints(name)
    saved={'dataset':name,'d':source['d'],'reference_indices':ri.tolist(),'inner':blocks,
        'unique_metrics':unique,'fingerprints':hashes,'complete':True,'outer_labels_used':False,
        'outer_scores_read':False,'goal_achieved':False,'scope':'dependent_inner_development_not_prediction'}
    saved['summary']=summarize_saved(saved)
    path.write_text(json.dumps(saved,separators=(',',':')),encoding='utf-8');return saved


def summary(reports):
    methods={}
    for m in METHODS:
        rows=[{'dataset':r['dataset'],**r['summary'][m]} for r in reports]
        methods[m]={'dataset_equal_five_minus_binary_pp':float(np.mean([r['five_minus_binary_pp'] for r in rows])),
            'datasets_better':sum(r['five_minus_binary_pp']>1e-10 for r in rows),
            'datasets_worse':sum(r['five_minus_binary_pp']<-1e-10 for r in rows),
            'intermediate_models':sum(r['intermediate_models'] for r in rows),
            'folds_with_strictly_better_intermediate':sum(r['folds_with_strictly_better_intermediate'] for r in rows),
            'dataset_equal_optimistic_fold_oracle_extra_pp':float(np.mean([r['optimistic_fold_oracle_mean_extra_pp'] for r in rows])),
            'rows':rows}
    return {'datasets':len(reports),'inner_folds':sum(len(r['inner']) for r in reports),
        'unique_metrics':sum(r['unique_metrics'] for r in reports),'methods':methods,'goal_achieved':False,
        'independent_audit_required':True,'limits':['No outer predictions or comparisons.',
            'Per-fold oracle is optimistic by construction; five-recipe selection also reuses inner folds.',
            'Original source recipes were selected on these same folds; no unbiased validation-risk or superiority claim.']}


def main():
    threadpool_limits(1);FOLDER.mkdir(exist_ok=True);audit=verify(SOURCE);names=sorted(audit['artifact_hashes']);assert len(names)==15
    frozen=json.loads((ROOT/'research/certified-prefix-diagnostic-source-freeze.json').read_text())
    for rel,sha in frozen['fingerprints'].items():assert digest(ROOT/rel)==sha
    reports=[run(n,audit) for n in names];result=summary(reports);assert result['inner_folds']==44
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({m:{k:v for k,v in r.items() if k!='rows'} for m,r in result['methods'].items()},indent=2))


if __name__=='__main__':main()
