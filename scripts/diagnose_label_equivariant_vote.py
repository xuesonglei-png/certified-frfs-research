"""ALL15/44 fixed crossgroup metrics, inner-only common geometric tie rule."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from run_simplex_soft5_knn_pilot import ROOT,digest
from continue_soft_topk_knn_pilot import verify
from label_equivariant_vote import predict,independent_predict_vectorized
from run_crossgroup_soft5_diagnostic import METHODS

SOURCE=ROOT/'results/crossgroup-soft5-diagnostic'
FOLDER=ROOT/'results/label-equivariant-vote-diagnostic'


def balanced(pred,truth):
    pred,truth=np.asarray(pred),np.asarray(truth)
    return float(np.mean([np.mean(pred[truth==label]==label) for label in np.unique(truth)]))


def main():
    threadpool_limits(1);FOLDER.mkdir(exist_ok=True);audit=verify(SOURCE)
    frozen=json.loads((ROOT/'research/label-equivariant-vote-diagnostic-source-freeze.json').read_text())
    for rel,sha in frozen['fingerprints'].items():assert digest(ROOT/rel)==sha,rel
    reports=[];unique_models=0;tie_queries=0;changed_queries=0
    for name,sha in sorted(audit['artifact_hashes'].items()):
        path=SOURCE/(name+'.json');assert digest(path)==sha;source=json.loads(path.read_text())
        out=FOLDER/(name+'.json');hashes={**frozen['fingerprints'],path.relative_to(ROOT).as_posix():sha,
            (SOURCE/'complete-audit.json').relative_to(ROOT).as_posix():digest(SOURCE/'complete-audit.json')}
        if out.exists():
            saved=json.loads(out.read_text());assert saved['fingerprints']==hashes
            reports.append(saved);continue
        X,y=load_dataset(name);ri=np.array(source['reference_indices']);raw,labels=X[ri],y[ri];blocks=[]
        for fi,b in enumerate(source['inner']):
            ca,cb=np.array(b['reference_indices']),np.array(b['validation_indices']);assert not set(ca)&set(cb)
            A,B=preprocess_fit(raw[ca],raw[cb]);cache={};records=[];options={}
            for m in METHODS:
                options[m]=[]
                for i,op in enumerate(b['options'][m]):
                    metric=op['metric'];fs,w=metric['features'],metric['weights'];key=(tuple(fs),tuple(w))
                    if key not in cache:
                        D=cdist(B[:,fs]*w,A[:,fs]*w,'cityblock');fresh=predict(D,labels[ca])
                        assert fresh['neighbor_indices']==op['evaluation']['neighbor_indices']
                        np.testing.assert_allclose(fresh['probabilities'],op['evaluation']['probabilities'],atol=0,rtol=0)
                        independent=independent_predict_vectorized(A,labels[ca],B,fs,w)
                        assert fresh['predictions']==independent
                        tied=np.array(fresh['tied_queries']);old=np.array(op['evaluation']['predictions'])
                        assert np.array_equal(np.array(fresh['predictions'])[~tied],old[~tied])
                        cache[key]=len(records)
                        records.append({'metric':{'features':fs,'weights':w},'report':fresh,
                            'BA':balanced(fresh['predictions'],labels[cb]),
                            'old_BA':op['evaluation']['metrics']['clean_BA'],
                            'balanced_logloss':op['evaluation']['metrics']['balanced_logloss'],
                            'changed_queries':int(np.sum(np.array(fresh['predictions'])!=old)),
                            'independent_coordinate_vote_replay_passed':True})
                    options[m].append({'target_index':i,'record_id':cache[key]})
            blocks.append({'reference_indices':ca.tolist(),'validation_indices':cb.tolist(),'records':records,'options':options})
            print(name,'inner',fi+1,'common geometric vote rule independently replayed',len(records),flush=True)
        selections={};old_indices=source['selection']['selected']
        for m in METHODS:
            rows=[]
            for i in range(len(blocks[0]['options'][m])):
                r=[b['records'][b['options'][m][i]['record_id']] for b in blocks]
                rows.append({'target_index':i,'mean_BA':float(np.mean([v['BA'] for v in r])),
                    'mean_old_BA':float(np.mean([v['old_BA'] for v in r])),
                    'mean_logloss':float(np.mean([v['balanced_logloss'] for v in r]))})
            choice=min(range(len(rows)),key=lambda i:(-round(rows[i]['mean_BA'],12),round(rows[i]['mean_logloss'],12),i))
            selections[m]={'rows':rows,'selected':choice,'old_selected':old_indices[m],
                'fixed_old_recipe_gain_pp':100*(rows[old_indices[m]]['mean_BA']-rows[old_indices[m]]['mean_old_BA']),
                'reselected_inner_gain_pp':100*(rows[choice]['mean_BA']-rows[old_indices[m]]['mean_old_BA'])}
        saved={'dataset':name,'reference_indices':ri.tolist(),'fingerprints':hashes,'inner':blocks,'selection':selections,
            'source_sha256':sha,'source_audit_sha256':digest(SOURCE/'complete-audit.json'),'complete':True,
            'goal_achieved':False,'new_outer_evaluation_performed':False,'independent_coordinate_vote_replay_passed':True,
            'scope':'all15_shared_geometric_tie_rule_inner_development_not_new_feature_fitting_or_test_advantage'}
        out.write_text(json.dumps(saved,separators=(',',':')),encoding='utf-8');reports.append(saved)
    for saved in reports:
        for b in saved['inner']:
            unique_models+=len(b['records']);tie_queries+=sum(sum(r['report']['tied_queries']) for r in b['records'])
            changed_queries+=sum(r['changed_queries'] for r in b['records'])
    means={m:{k:float(np.mean([s['selection'][m][k] for s in reports])) for k in
        ('fixed_old_recipe_gain_pp','reselected_inner_gain_pp')} for m in METHODS}
    assert len(reports)==15 and sum(len(s['inner']) for s in reports)==44
    summary={'means':means,'counts':{'datasets':15,'inner_folds':44,'unique_model_predictions':unique_models,
        'tie_queries':tie_queries,'changed_queries':changed_queries},'artifact_hashes':{s['dataset']:digest(FOLDER/(s['dataset']+'.json')) for s in reports},
        'new_outer_evaluation_performed':False,'independent_coordinate_vote_replay_passed':True,'goal_achieved':False,
        'scope':'all15_common_geometric_tie_rule_inner_only_not_test_or_population_advantage',
        'limitations':['Frozen metric anchors and inner recipes were already exposed development.',
            'No nativeFGS120 geometric-rule comparison yet; cannot claim proposed superiority.',
            'Reselected inner gains include optimization on the same inner validation, not generalization evidence.',
            'Three-friend majority certificates carry over because all strict 5NN majorities are identical for every table realization.']}
    (FOLDER/'complete.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
