"""Independent source/pruning/count/selection/query replay for ALL15."""
import json
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_thinned_atomic_classifier_v2 import ROOT,FOLDER,SOURCE,POOL,METHODS,REFINED,SOURCE_ONLY,EXACT,digest
from continue_soft_topk_knn_pilot import verify
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from independent_thinned_atomic_pruning import check_thinned_atomic_pruning as check_pruning
from audit_simplex_soft5_knn_pilot import check_stage,check_metric
from exact_count_anchor_v2 import independent_exact_count_policies as policies
from audit_certified_metric_knn_pool import independent_evaluate,query_certificate
from audit_forest_compact_transfer import equivalent
from simplex_soft5_knn_stage import GRID
from review_box_iterations import paired
from run_nested_joint_path import holm


def original_metric(parent,block,method,outer=False):
    if outer: return block['results'][method]['metric']
    return block['options'][method][parent['selection']['selected'][method]]['metric']


def replay_pruning(A,y,block,counts):
    keys=[]
    for r in block['pruning_records']:
        assert r['fit']['fraction']==.5 and r['fit']['pool']==4
        assert r['fit']['receiver_pool']==2 and r['fit']['max_checks']==48
        key=(tuple(r['initial_features']),tuple(r['initial_weights'])); assert key not in keys; keys.append(key)
        counts['independent_pruning_candidate_votes']+=check_pruning(A,y,r['initial_features'],r['initial_weights'],r['fit'])
        assert r['features']==r['fit']['features']; equivalent(r['weights'],r['fit']['weights'])
    return keys


def check_refinement(metric,base,variant,records,method):
    assert metric['variant']==variant
    if variant==0:
        assert metric['features']==base['features'] and metric['pruning_id'] is None and metric['guard_reference'] is None
        equivalent(metric['weights'],base['weights'])
    else:
        r=records[metric['pruning_id']]
        assert r['initial_features']==base['features']; equivalent(r['initial_weights'],base['weights'])
        assert metric['features']==r['features']; equivalent(metric['weights'],r['weights'])
        assert metric['guard_reference']==r['fit']['guard_reference']
        if method=='proposed': assert metric['guard_reference']=='original_full'


def replay_final(A,y,B,truth,result,counts):
    fs,w=result['metric']['features'],result['metric']['weights']
    report=independent_evaluate(A,y,B,truth,fs,w)
    cert=query_certificate(A,y,B,np.array(report['predictions']),fs,w); equivalent(cert,result['query_certificate'])
    correct=(np.array(report['predictions'])==truth)&np.array(cert['certified'])
    report['metrics']['query_certified_BA']=float(np.mean([np.mean(correct[truth==c]) for c in np.unique(truth)]))
    assert len(result['perturbation_evaluations'])==16; noisy=[]
    for draw,stored in enumerate(result['perturbation_evaluations']):
        rng=np.random.default_rng(20261060+draw)
        na=np.clip(A+rng.uniform(-.01,.01,A.shape),0,1); nb=np.clip(B+rng.uniform(-.01,.01,B.shape),0,1)
        fresh=independent_evaluate(na,y,nb,truth,fs,w); equivalent(fresh,stored)
        assert not np.any((np.array(fresh['predictions'])!=report['predictions'])&np.array(cert['certified']))
        noisy.append(fresh['metrics']['clean_BA']); counts['perturbation_predictions']+=1
    report['metrics']['mean_perturbed_BA']=float(np.mean(noisy)); equivalent(report,result['evaluation'])
    counts['outer_predictions']+=1
    return report['metrics']


def audit(name):
    threadpool_limits(1); path=FOLDER/(name+'.json'); saved=json.loads(path.read_text())
    assert saved['control_recipe_version']==2
    assert saved['complete'] and not saved['query_labels_used_for_selection']
    if 'raw_reuse_provenance' in saved:
        from reuse_thinned_atomic_classifier_v2 import validate_reused_artifact
        validate_reused_artifact(name,saved)
    for rel,sha in saved['fingerprints'].items(): assert digest(ROOT/rel)==sha
    pa=verify(SOURCE); ma=verify(POOL)
    assert pa['artifact_hashes'][name]==digest(SOURCE/(name+'.json')) and ma['artifact_hashes'][name]==digest(POOL/(name+'.json'))
    parent=json.loads((SOURCE/(name+'.json')).read_text()); raw_parent=json.loads((POOL/(name+'.json')).read_text())
    assert saved['reference_indices']==parent['reference_indices']==raw_parent['reference_indices']
    assert saved['validation_indices']==parent['validation_indices']==raw_parent['validation_indices']
    X,y=load_dataset(name); ri,vi=np.array(saved['reference_indices']),np.array(saved['validation_indices'])
    assert not set(ri)&set(vi); raw,labels=X[ri],y[ri]; counts=Counter(datasets=1)
    combined,selected=saved['selection']['options'],saved['selection']['selected']
    assert set(combined)==set(selected)==set(METHODS)
    for m in METHODS:
        width=2 if m in REFINED else (1 if m in SOURCE_ONLY else ((9 if m.startswith('fgs') else 6)*(len(GRID) if m in ('fgs_exact','sfrip_exact') else 1)))
        assert len(combined[m])==width
        for i,row in enumerate(combined[m]):
            folds=[b['options'][m][i] for b in saved['inner']]
            equivalent(row['mean_clean_BA'],float(np.mean([r['evaluation']['metrics']['clean_BA'] for r in folds])))
            equivalent(row['mean_balanced_logloss'],float(np.mean([r['evaluation']['metrics']['balanced_logloss'] for r in folds])))
            equivalent(row['mean_actual_count'],float(np.mean([len(r['metric']['features']) for r in folds])))
            for key in ('variant','selector_index','target_index'):
                if key in folds[0]: assert row[key]==folds[0][key] and all(f[key]==row[key] for f in folds)
        chosen=min(range(len(combined[m])),key=lambda i:(-round(combined[m][i]['mean_clean_BA'],12),
            round(combined[m][i]['mean_balanced_logloss'],12),combined[m][i]['mean_actual_count'],i))
        assert selected[m]==chosen
    variant=combined['proposed'][selected['proposed']]['variant']
    assert len(saved['inner'])==len(parent['inner'])==len(raw_parent['candidate_reference_blocks'])-1
    for bi,(block,source,raw_block) in enumerate(zip(saved['inner'],parent['inner'],raw_parent['candidate_reference_blocks'][:-1],strict=True)):
        assert block['reference_indices']==source['reference_indices']==raw_block['reference_indices']
        assert block['validation_indices']==source['validation_indices']==raw_block['validation_indices']
        ca,cb=np.array(block['reference_indices']),np.array(block['validation_indices']); assert not set(ca)&set(cb)
        A,B=preprocess_fit(raw[ca],raw[cb]); replay_pruning(A,labels[ca],block,counts)
        for m,old in REFINED.items():
            assert len(block['options'][m])==2; base=original_metric(parent,source,old)
            for v,row in enumerate(block['options'][m]):
                assert row['variant']==v; check_refinement(row['metric'],base,v,block['pruning_records'],m)
        for m,old in SOURCE_ONLY.items():
            assert len(block['options'][m])==1
            check_refinement(block['options'][m][0]['metric'],original_metric(parent,source,old),0,block['pruning_records'],m)
        count=len(block['options']['proposed'][variant]['metric']['features']); assert block['exact_proposed_count']==count
        stage=block['exact_stage']; check_stage(A,labels[ca],stage,counts); bases=policies(A,raw_block,count=count)
        for m,key in EXACT.items():
            weighted=m in ('fgs_exact','sfrip_exact'); step=len(GRID) if weighted else 1
            assert len(block['options'][m])==len(bases[key])*step
            for i,row in enumerate(block['options'][m]):
                si,ti=i//step,i%step; assert row['selector_index']==si and row['target_index']==ti
                check_metric(row['metric'],bases[key][si],stage,key,ti,counts)
                assert len(row['metric']['features'])==count
        for m,rows in block['options'].items():
            assert m in METHODS
            for row in rows:
                metric=row['metric']; fresh=independent_evaluate(A,labels[ca],B,labels[cb],metric['features'],metric['weights'])
                equivalent(fresh,row['evaluation']); counts['inner_predictions']+=1
        counts['inner_folds']+=1; print(name,'inner',bi+1,'pruning/count/choice/query replayed',flush=True)
    A,B=preprocess_fit(raw,X[vi]); outer=saved['outer']; replay_pruning(A,labels,outer,counts)
    check_stage(A,labels,outer['exact_stage'],counts); count=len(outer['results']['proposed']['metric']['features'])
    assert outer['exact_proposed_count']==count; bases=policies(A,raw_parent['candidate_reference_blocks'][-1],count=count)
    metrics={}; sizes={}
    for m in METHODS:
        result=outer['results'][m]; assert result['option_index']==selected[m]; metric=result['metric']
        if m in REFINED:
            check_refinement(metric,original_metric(parent,parent['outer'],REFINED[m],True),combined[m][selected[m]]['variant'],outer['pruning_records'],m)
        elif m in SOURCE_ONLY:
            check_refinement(metric,original_metric(parent,parent['outer'],SOURCE_ONLY[m],True),0,outer['pruning_records'],m)
        else:
            row=combined[m][selected[m]]; key=EXACT[m]
            assert result['selector_index']==row['selector_index'] and result['target_index']==row['target_index']
            check_metric(metric,bases[key][row['selector_index']],outer['exact_stage'],key,row['target_index'],counts)
            assert len(metric['features'])==count
        metrics[m]=replay_final(A,labels,B,y[vi],result,counts); sizes[m]=len(metric['features'])
        if m in SOURCE_ONLY: equivalent(result['evaluation'],parent['outer']['results'][SOURCE_ONLY[m]]['evaluation'])
    row={'dataset':name,'d':saved['d'],'actual_counts':sizes,'removed':1-count/saved['d'],'metrics':metrics,
         'positive_weight_counts':{m:sum(w>0 for w in outer['results'][m]['metric']['weights']) for m in METHODS}}
    print(name,'COMPLETE independently audited postmetric classifier',flush=True)
    return {'dataset':name,'row':row,'counts':dict(counts),'sha256':digest(path)}


if __name__=='__main__':
    summary=json.loads((FOLDER/'complete.json').read_text()); reports=[]; counts=Counter()
    with ProcessPoolExecutor(max_workers=1) as executor:
        for f in as_completed([executor.submit(audit,r['dataset']) for r in summary['rows']]):
            r=f.result(); reports.append(r); counts.update(r['counts'])
    rows=sorted([r['row'] for r in reports],key=lambda r:r['dataset']); equivalent(rows,summary['rows'])
    assert counts['datasets']==15 and counts['inner_folds']==44
    means={m:{k:float(np.mean([r['metrics'][m][k] for r in rows])) for k in rows[0]['metrics'][m]} for m in METHODS}
    equivalent(means,summary['means'])
    contrasts={m:paired([r['metrics']['proposed']['clean_BA']-r['metrics'][m]['clean_BA'] for r in rows]) for m in METHODS if m!='proposed'}
    equivalent(contrasts,summary['contrasts_clean_BA']); primary=('fgs_refined','sfrip_refined')
    equivalent(holm({m:contrasts[m]['sign_flip_two_sided_p'] for m in primary}),summary['primary_descriptive_holm'])
    removed=float(np.mean([r['removed'] for r in rows])); equivalent(removed,summary['mean_removed'])
    inherited=json.loads((SOURCE/'complete.json').read_text())['counts']['target_optimizer_failures']
    assert summary['counts']['inherited_optimizer_failures']==inherited==7
    assert summary['counts']['new_exact_optimizer_failures']==counts['target_optimizer_failures']
    assert summary['counts']['pruning_checks']==counts['independent_pruning_candidate_votes']
    gate=all(contrasts[m]['mean_pp']>=.5 and contrasts[m]['wins']>=10 for m in primary) and removed>=.1 and contrasts['full_source']['mean_pp']>=0 and inherited+counts['target_optimizer_failures']==0
    assert summary['pilot_gate_passed']==bool(gate)
    output={'passed':True,'counts':dict(counts),'artifact_hashes':{r['dataset']:r['sha256'] for r in reports},
        'summary_sha256':digest(FOLDER/'complete.json'),'auditor_sha256':digest(Path(__file__)),'goal_achieved':False,
        'scope':'all15_independent_source_pruning_guard_training_vote_count_choice_scalar_queries_and_noise_not_confirmation'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(output,indent=2),encoding='utf-8'); print(output,flush=True)
