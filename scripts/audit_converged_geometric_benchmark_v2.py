"""Independent supports, target objectives, guards, choices, queries/noise."""
import json
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_converged_geometric_benchmark import ROOT,FOLDER,SOURCE,PATHS,POOL,METHODS,REFINED,WEIGHTED,MATCHED,PRIMARY,GRID,digest
from exact_count_anchor_v2 import independent_exact_count_policies
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from audit_converged_crossgroup_stage import check_stage,check_metric
from independent_geometric_evaluation import independent_evaluate,replay_final
from audit_forest_compact_transfer import equivalent
from review_box_iterations import paired
from run_nested_joint_path import holm


def independent_anchors(A,atomic,graph_block,pool_block,selected,outer=False):
    d=len(A[0]);count=atomic['exact_proposed_count'];result={}
    for m in REFINED:
        r=atomic['results'][m] if outer else atomic['options'][m][selected[m]]
        result[m]=[{'features':r['metric']['features'],'weights':r['metric']['weights'],
                    'origin':'frozen_atomic_v2_selected_metric'}]
    graph=[r['path'] for r in graph_block['paths']];sfrip=pool_block['paths']['sfrip2025']
    for prefix,paths,width in [('fgs',graph,120),('sfrip',sfrip,6)]:
        assert len(paths)==width
        plain=[];learned=[]
        for p in paths:
            support=sorted(p['native']['features'])
            plain.append({'features':support,'weights':[1. for _ in support],'origin':'native_uniform'})
            learned.append({'features':support,'weights':[float(d**.5/len(support)) for _ in support],
                            'origin':'native_fixed_mass_uniform'})
        result[f'{prefix}_native{width}']=plain;result[f'{prefix}_weighted_native{width}']=learned
    old=independent_exact_count_policies(A,{'pool':[],'recipes':[],'paths':{'fgs2023':graph,'sfrip2025':sfrip}},count)
    for m,k in [('fgs_weighted_count120','fgs_matched'),('fgs_uniform_count120','fgs_uniform_matched'),
                ('sfrip_weighted_count6','sfrip_matched'),('sfrip_uniform_count6','sfrip_uniform_matched')]:result[m]=old[k]
    result['full_uniform']=[{'features':list(range(d)),'weights':[1/d**.5 for _ in range(d)],'origin':'uniform_full'}]
    assert set(result)==set(METHODS);return result


def check_options(A,y,block,bases,counts):
    equivalent(block['bases'],bases);check_stage(A,y,block['stage'],counts)
    referenced=set()
    for m in METHODS:
        width=len(GRID) if m in WEIGHTED else 1
        assert len(block['options'][m])==width*len(bases[m])
        for i,op in enumerate(block['options'][m]):
            ci,ti=divmod(i,width);assert (ci,ti)==(op['config_index'],op['target_index'])
            base=bases[m][ci];metric=op['metric'];weighted=m in WEIGHTED and bool(base['features'])
            surrogate='proposed' if m=='proposed' else 'full' if weighted else 'unweighted_control'
            check_metric(metric,base,block['stage'],surrogate,ti if weighted else 0,counts)
            rid=op['record_id'];assert 0<=rid<len(block['records']);referenced.add(rid)
            equivalent({k:metric[k] for k in ('features','weights')},
                       {k:block['records'][rid]['metric'][k] for k in ('features','weights')})
    assert referenced==set(range(len(block['records'])))


def independent_selection(blocks,m):
    rows=[]
    width=len(GRID) if m in WEIGHTED else 1
    for i in range(len(blocks[0]['options'][m])):
        scores=[];sizes=[]
        for block in blocks:
            record=block['records'][block['options'][m][i]['record_id']]
            scores.append(record['evaluation']['metrics']);sizes.append(len(record['metric']['features']))
        ci,ti=divmod(i,width)
        rows.append({'config_index':ci,'target_index':ti,'mean_BA':float(np.mean([r['clean_BA'] for r in scores])),
            'mean_logloss':float(np.mean([r['balanced_logloss'] for r in scores])),'mean_count':float(np.mean(sizes))})
    chosen=min(range(len(rows)),key=lambda i:(-round(rows[i]['mean_BA'],12),round(rows[i]['mean_logloss'],12),rows[i]['mean_count'],i))
    return {'options':rows,'selected_index':chosen}


def audit(name):
    threadpool_limits(1);path=FOLDER/(name+'.json');saved=json.loads(path.read_text())
    assert saved['complete'] and not saved['query_labels_used_for_selection'];counts=Counter(datasets=1)
    for rel,sha in saved['fingerprints'].items():assert digest(ROOT/rel)==sha,rel
    source=json.loads((SOURCE/(name+'.json')).read_text());graph=json.loads((PATHS/(name+'.json')).read_text())
    pool=json.loads((POOL/(name+'.json')).read_text())
    for folder,blob in [(SOURCE,source),(PATHS,graph),(POOL,pool)]:
        a=json.loads((folder/'complete-audit.json').read_text());assert a['passed']
        assert a['summary_sha256']==digest(folder/'complete.json') and a['artifact_hashes'][name]==digest(folder/(name+'.json'))
        assert blob['reference_indices']==saved['reference_indices'] and blob['validation_indices']==saved['validation_indices']
    X,y=load_dataset(name);ri,vi=np.array(saved['reference_indices']),np.array(saved['validation_indices'])
    assert not set(ri)&set(vi);raw,labels=X[ri],y[ri];selected=source['selection']['selected']
    for fi,(b,s,g,p) in enumerate(zip(saved['inner'],source['inner'],graph['inner'],pool['candidate_reference_blocks'][:-1],strict=True)):
        for other in (s,g,p):
            assert other['reference_indices']==b['reference_indices'] and other['validation_indices']==b['validation_indices']
        ca,cb=np.array(b['reference_indices']),np.array(b['validation_indices']);assert not set(ca)&set(cb)
        A,B=preprocess_fit(raw[ca],raw[cb]);bases=independent_anchors(A,s,g,p,selected)
        assert b['count']==s['exact_proposed_count'];check_options(A,labels[ca],b,bases,counts)
        for r in b['records']:
            fresh=independent_evaluate(A,labels[ca],B,labels[cb],r['metric']);equivalent(fresh,r['evaluation'])
            counts['independent_inner_predictions']+=1
        counts['inner_folds']+=1;print(name,'inner',fi+1,'independent target/support/guard/vote audit',flush=True)
    choice={m:independent_selection(saved['inner'],m) for m in METHODS};equivalent(choice,saved['selection'])
    A,B=preprocess_fit(raw,X[vi]);outer=saved['outer'];bases=independent_anchors(A,source['outer'],graph['outer'],pool['candidate_reference_blocks'][-1],selected,True)
    equivalent(bases,outer['bases']);check_stage(A,labels,outer['stage'],counts)
    metrics={};sizes={};positive={}
    for m in METHODS:
        r=outer['results'][m];recipe=choice[m]['options'][choice[m]['selected_index']]
        assert r['option_index']==choice[m]['selected_index']
        equivalent(recipe,{k:r[k] for k in recipe});base=bases[m][recipe['config_index']]
        weighted=m in WEIGHTED and bool(base['features']);surrogate='proposed' if m=='proposed' else 'full' if weighted else 'unweighted_control'
        check_metric(r['metric'],base,outer['stage'],surrogate,recipe['target_index'] if weighted else 0,counts)
        metrics[m]=replay_final(A,labels,B,y[vi],r,counts);sizes[m]=len(r['metric']['features'])
        positive[m]=sum(w>0 for w in r['metric']['weights'])
    assert outer['count']==sizes['proposed']==source['outer']['exact_proposed_count']
    assert all(sizes[m]==sizes['proposed'] for m in MATCHED)
    counts['proposed_fallback_options']=sum(op['metric']['guard_fallback'] for b in saved['inner'] for op in b['options']['proposed'])+outer['results']['proposed']['metric']['guard_fallback']
    row={'dataset':name,'d':saved['d'],'removed':1-sizes['proposed']/saved['d'],
         'actual_counts':sizes,'positive_weight_counts':positive,'metrics':metrics}
    print(name,'COMPLETE independent audit',flush=True)
    return {'row':row,'counts':dict(counts),'sha256':digest(path)}


def main():
    summary=json.loads((FOLDER/'complete.json').read_text());reports=[];counts=Counter()
    with ProcessPoolExecutor(max_workers=2) as executor:
        for f in as_completed([executor.submit(audit,r['dataset']) for r in summary['rows']]):
            r=f.result();reports.append(r);counts.update(r['counts'])
    rows=sorted([r['row'] for r in reports],key=lambda r:r['dataset']);equivalent(rows,summary['rows'])
    assert counts['datasets']==15 and counts['inner_folds']==44
    means={m:{k:float(np.mean([r['metrics'][m][k] for r in rows])) for k in rows[0]['metrics'][m]} for m in METHODS}
    equivalent(means,summary['means'])
    contrasts={m:paired([r['metrics']['proposed']['clean_BA']-r['metrics'][m]['clean_BA'] for r in rows]) for m in METHODS if m!='proposed'}
    equivalent(contrasts,summary['contrasts_clean_BA']);corrected=holm({m:contrasts[m]['sign_flip_two_sided_p'] for m in PRIMARY})
    equivalent(corrected,summary['primary_descriptive_holm']);removed=float(np.mean([r['removed'] for r in rows]));equivalent(removed,summary['mean_removed'])
    gate=all(contrasts[m]['mean_pp']>=.5 and contrasts[m]['wins']>=10 and contrasts[m]['bootstrap95_pp'][0]>0
             and corrected[m]<.05 for m in PRIMARY) and removed>=.1 and contrasts['full_uniform']['mean_pp']>=0 and counts['target_optimizer_failures']==0
    assert summary['pilot_gate_passed']==bool(gate)
    assert summary['counts']['new_target_optimizer_failures']==counts['target_optimizer_failures']
    assert summary['counts']['proposed_fallback_options']==counts['proposed_fallback_options']
    inherited=json.loads((SOURCE/'complete.json').read_text())['counts']
    assert summary['counts']['upstream_optimizer_failures_preserved']==inherited['inherited_optimizer_failures']+inherited['new_exact_optimizer_failures']
    output={'passed':True,'counts':dict(counts),'artifact_hashes':{r['row']['dataset']:r['sha256'] for r in reports},
        'summary_sha256':digest(FOLDER/'complete.json'),'auditor_sha256':digest(Path(__file__)),
        'goal_achieved':False,'scope':'independent_all15_refinement_objective_guard_prediction_query_noise_not_new_confirmation'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(output,indent=2),encoding='utf-8');print(json.dumps(output,indent=2),flush=True)


if __name__=='__main__':main()
