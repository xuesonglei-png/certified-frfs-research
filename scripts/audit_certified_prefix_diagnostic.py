"""Independently reconstruct every scored prefix, inner vote and recipe choice."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_certified_prefix_diagnostic import ROOT,SOURCE,FOLDER,METHODS,digest,fingerprints,summarize_saved,summary
from continue_soft_topk_knn_pilot import verify
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from audit_certified_metric_knn_pool import independent_evaluate
from audit_forest_compact_transfer import equivalent


def main():
    threadpool_limits(1);source_audit=verify(SOURCE);names=sorted(source_audit['artifact_hashes']);reports=[];replays=0
    for name in names:
        path=FOLDER/(name+'.json');r=json.loads(path.read_text());src=json.loads((SOURCE/(name+'.json')).read_text())
        assert r['complete'] and not r['outer_labels_used'] and not r['outer_scores_read'] and r['fingerprints']==fingerprints(name)
        assert digest(SOURCE/(name+'.json'))==source_audit['artifact_hashes'][name]
        assert r['reference_indices']==src['reference_indices'];X,y=load_dataset(name);ri=np.array(r['reference_indices']);raw,labels=X[ri],y[ri]
        assert len(r['inner'])==len(src['inner']);unique=0
        for block,prior in zip(r['inner'],src['inner'],strict=True):
            ca,cb=np.array(block['reference_indices']),np.array(block['validation_indices'])
            assert block['reference_indices']==prior['reference_indices'] and block['validation_indices']==prior['validation_indices']
            assert not set(ca)&set(cb);A,B=preprocess_fit(raw[ca],raw[cb]);cache={};assert set(block['methods'])==set(METHODS)
            for m in METHODS:
                p=block['methods'][m];pid=prior['options'][m][1]['metric']['pruning_id'];assert p['source_pruning_id']==pid
                fit=prior['pruning_records'][pid]['fit'];assert fit['initial_guard']['accepted']
                if m=='proposed':assert fit['guard_reference']=='original_full'
                fs,w=fit['initial_features'],fit['initial_weights'];path_metrics=[(fs,w)]
                for chosen in fit['chosen_ids']:
                    point=fit['checks'][chosen];assert point['id']==chosen and point['eligible'] and point['accepted']
                    assert point['features_before']==fs and point['weights_before']==w
                    assert len(point['features'])==len(fs)-1 and set(point['features'])<set(fs)
                    fs,w=point['features'],point['weights'];path_metrics.append((fs,w))
                assert fs==fit['features'] and w==fit['weights'];assert len(p['prefixes'])==len(path_metrics)
                for step,(saved,(fs,w)) in enumerate(zip(p['prefixes'],path_metrics,strict=True)):
                    metric=saved['metric'];assert metric=={'features':fs,'weights':w,'path_step':step,'guard_reference':fit['guard_reference']}
                    key=(tuple(fs),tuple(w))
                    if key not in cache:
                        cache[key]=independent_evaluate(A,labels[ca],B,labels[cb],fs,w);replays+=1
                    equivalent(saved['evaluation'],cache[key])
                equivalent(p['prefixes'][0]['evaluation'],prior['options'][m][0]['evaluation'])
                equivalent(p['prefixes'][-1]['evaluation'],prior['options'][m][1]['evaluation'])
            unique+=len(cache)
        assert unique==r['unique_metrics']
        # Reconstruct selections independently of the production criterion.
        for m in METHODS:
            for kind,progress in [('binary',(0.,1.)),('five',(0.,.25,.5,.75,1.))]:
                rows=[]
                for fraction in progress:
                    picked=[]
                    for b in r['inner']:
                        points=b['methods'][m]['prefixes'];index=int(np.ceil(fraction*(len(points)-1)));picked.append(points[index])
                    rows.append({'progress':fraction,'mean_BA':float(np.mean([p['evaluation']['metrics']['clean_BA'] for p in picked])),
                        'mean_logloss':float(np.mean([p['evaluation']['metrics']['balanced_logloss'] for p in picked])),
                        'mean_count':float(np.mean([len(p['metric']['features']) for p in picked]))})
                best=min(range(len(rows)),key=lambda i:(-round(rows[i]['mean_BA'],12),round(rows[i]['mean_logloss'],12),rows[i]['mean_count'],i))
                equivalent(r['summary'][m][kind],{'rows':rows,'selected':best})
        equivalent(r['summary'],summarize_saved(r));reports.append(r)
        print(name,'all accepted inner prefixes independently replayed',unique,'metrics',flush=True)
    stored=json.loads((FOLDER/'complete.json').read_text());equivalent(stored,summary(reports))
    assert stored['datasets']==15 and stored['inner_folds']==44 and stored['unique_metrics']==replays
    output={'passed':True,'datasets':15,'inner_folds':44,'independent_prefix_evaluations':replays,
        'artifact_hashes':{n:digest(FOLDER/(n+'.json')) for n in names},'summary_sha256':digest(FOLDER/'complete.json'),
        'source_audit_sha256':digest(SOURCE/'complete-audit.json'),'auditor_sha256':digest(Path(__file__)),
        'goal_achieved':False,'scope':'independent_prefix_continuity_votes_choices; source_guard_math_reused_from_full_audit;no_outer_scoring'}
    (FOLDER/'complete-audit.json').write_text(json.dumps(output,indent=2),encoding='utf-8');print(output,flush=True)


if __name__=='__main__':main()
