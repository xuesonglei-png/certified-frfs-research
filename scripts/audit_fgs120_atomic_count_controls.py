"""Independent common-weighted FGS120/count/inner choice and outer/noise audit."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_fgs120_atomic_count_controls import ROOT,FOLDER,SOURCE,PATHS,METHODS,digest
from exact_count_anchor_v2 import independent_exact_count_policies
from simplex_soft5_knn_stage import GRID
from audit_simplex_soft5_knn_pilot import check_stage,check_metric
from audit_certified_metric_knn_pool import independent_evaluate
from audit_atomic_pruning_classifier_v2 import replay_final
from audit_forest_compact_transfer import equivalent
from continue_soft_topk_knn_pilot import verify
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from review_box_iterations import paired
from run_nested_joint_path import holm


def independent_bases(A,paths,count):
    raw={'pool':[],'recipes':[],'paths':{'fgs2023':[r['path'] for r in paths],'sfrip2025':[]}}
    bases=independent_exact_count_policies(A,raw,count)
    return {METHODS[0]:bases['fgs_matched'],METHODS[1]:bases['fgs_uniform_matched']}


def audit(name):
    threadpool_limits(1);path=FOLDER/(name+'.json');saved=json.loads(path.read_text());counts=Counter(datasets=1)
    assert saved['complete'] and not saved['query_labels_used_for_selection']
    for rel,sha in saved['fingerprints'].items():assert digest(ROOT/rel)==sha,rel
    for folder in (SOURCE,PATHS):assert verify(folder)['artifact_hashes'][name]==digest(folder/(name+'.json'))
    source=json.loads((SOURCE/(name+'.json')).read_text());paths=json.loads((PATHS/(name+'.json')).read_text())
    assert saved['reference_indices']==source['reference_indices']==paths['reference_indices']
    assert saved['validation_indices']==source['validation_indices']==paths['validation_indices']
    X,y=load_dataset(name);ri,vi=np.array(saved['reference_indices']),np.array(saved['validation_indices']);assert not set(ri)&set(vi)
    raw,labels=X[ri],y[ri]
    for fi,(block,old,pblock) in enumerate(zip(saved['inner'],source['inner'],paths['inner'],strict=True)):
        assert block['reference_indices']==old['reference_indices']==pblock['reference_indices']
        assert block['validation_indices']==old['validation_indices']==pblock['validation_indices']
        ca,cb=np.array(block['reference_indices']),np.array(block['validation_indices']);assert not set(ca)&set(cb)
        A,B=preprocess_fit(raw[ca],raw[cb]);count=block['count'];assert count==old['exact_proposed_count']
        anchors=independent_bases(A,pblock['paths'],count);check_stage(A,labels[ca],block['stage'],counts)
        lookup=set()
        for record in block['records']:
            metric=record['metric'];key=(tuple(metric['features']),tuple(metric['weights']));assert key not in lookup;lookup.add(key)
            equivalent(independent_evaluate(A,labels[ca],B,labels[cb],metric['features'],metric['weights']),record['evaluation'])
            counts['inner_unique_predictions']+=1
        assert set(block['options'])==set(METHODS)
        for method in METHODS:
            step=len(GRID) if method==METHODS[0] else 1
            assert len(anchors[method])==120 and len(block['options'][method])==120*step
            for i,option in enumerate(block['options'][method]):
                ci,ti=i//step,i%step;assert option['config_index']==ci and option['target_index']==ti
                metric=option['metric'];check_metric(metric,anchors[method][ci],block['stage'],
                    'fgs_matched' if step>1 else 'fgs_uniform_matched',ti,counts)
                record=block['records'][option['record_id']]
                assert metric['features']==record['metric']['features'] and metric['weights']==record['metric']['weights']
                assert len(metric['features'])==count;counts['inner_options']+=1
        counts['inner_folds']+=1;print(name,'inner',fi+1,'FGS120 common learner/count independently audited',flush=True)
    for method in METHODS:
        rows=saved['selection'][method]['options'];assert len(rows)==len(saved['inner'][0]['options'][method])
        step=len(GRID) if method==METHODS[0] else 1
        for i,row in enumerate(rows):
            assert row['config_index']==i//step and row['target_index']==i%step
            values=[b['records'][b['options'][method][i]['record_id']]['evaluation']['metrics'] for b in saved['inner']]
            equivalent(row['mean_BA'],float(np.mean([v['clean_BA'] for v in values])))
            equivalent(row['mean_logloss'],float(np.mean([v['balanced_logloss'] for v in values])))
        chosen=min(range(len(rows)),key=lambda i:(-round(rows[i]['mean_BA'],12),round(rows[i]['mean_logloss'],12),i))
        assert saved['selection'][method]['selected_index']==chosen
    A,B=preprocess_fit(raw,X[vi]);outer=saved['outer'];assert outer['count']==source['outer']['exact_proposed_count']
    anchors=independent_bases(A,paths['outer']['paths'],outer['count']);check_stage(A,labels,outer['stage'],counts)
    for method in METHODS:
        result=outer['results'][method];selection=saved['selection'][method]
        assert result['option_index']==selection['selected_index'];op=selection['options'][result['option_index']]
        assert result['config_index']==op['config_index'] and result['target_index']==op['target_index']
        check_metric(result['metric'],anchors[method][op['config_index']],outer['stage'],
            'fgs_matched' if method==METHODS[0] else 'fgs_uniform_matched',op['target_index'],counts)
        assert len(result['metric']['features'])==outer['count'];replay_final(A,labels,B,y[vi],result,counts)
    row={'dataset':name,'count':outer['count'],'metrics':{m:outer['results'][m]['evaluation']['metrics'] for m in METHODS},
        'proposed_metrics':source['outer']['results']['proposed']['evaluation']['metrics']}
    return {'dataset':name,'row':row,'sha256':digest(path),'counts':dict(counts)}


if __name__=='__main__':
    summary=json.loads((FOLDER/'complete.json').read_text());reports=[];counts=Counter()
    for row in summary['rows']:
        report=audit(row['dataset']);reports.append(report);counts.update(report['counts'])
    rows=[r['row'] for r in reports];equivalent(rows,summary['rows'])
    means={m:float(np.mean([r['metrics'][m]['clean_BA'] for r in rows])) for m in METHODS};equivalent(means,summary['means'])
    contrasts={m:paired([r['proposed_metrics']['clean_BA']-r['metrics'][m]['clean_BA'] for r in rows]) for m in METHODS}
    equivalent(contrasts,summary['proposed_minus_controls']);equivalent(holm({m:contrasts[m]['sign_flip_two_sided_p'] for m in METHODS}),summary['descriptive_holm'])
    assert summary['target_optimizer_failures']==counts['target_optimizer_failures']
    assert counts['datasets']==15 and counts['inner_folds']==44
    result={'passed':True,'counts':dict(counts),'artifact_hashes':{r['dataset']:r['sha256'] for r in reports},
        'summary_sha256':digest(FOLDER/'complete.json'),'auditor_sha256':digest(Path(__file__)),
        'goal_achieved':False,'scope':'all15_common_weighted_FGS120_and_uniform_atomic_exact_counts_independent_targets_votes_choices_noise_not_confirmation'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(result,flush=True)
