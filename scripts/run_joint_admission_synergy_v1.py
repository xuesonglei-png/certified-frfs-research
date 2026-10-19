"""Fixed six-dataset, first-inner-training-only certificate synergy screen."""
import json,time,traceback
from pathlib import Path
from collections import Counter
import numpy as np
from threadpoolctl import threadpool_limits
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from joint_admission_synergy_v1 import JointAdmission
from diagnose_repair_bottlenecks_v7 import ROOT,NAMES,digest

FOLDER = ROOT/'results/joint-admission-synergy-v1'
ARMS = ('legacy','signed_only','expanded_only','joint')


def run(name,hashes):
    path=FOLDER/(name+'.json'); assert not path.exists()
    parent=json.loads((ROOT/'results/repair-relation-selector-v4'/(name+'.json')).read_text());assert parent['complete']
    panel=parent['rows'][0];block=panel['fit']['inner'][0]
    X,y=load_dataset(name);ri=np.array(panel['train_indices']);ii=np.array(block['fit_indices'])
    A,_=preprocess_fit(X[ri[ii]],X[ri[ii]]);labels=y[ri[ii]];guard=JointAdmission(A,labels)
    rows=[];counts=Counter();start=time.perf_counter()
    for family,part in block['families'].items():
        unique={}
        for recipe in part['candidates']:
            for c in recipe['trace']:
                ref=c['family_certificate']['dependency_certificate'].get('full_weights',c['weights'])
                unique[(tuple(c['features']),tuple(c['weights']),tuple(ref))]=c
        for c in unique.values():
            value=guard.candidate(c);rows.append({'family':family,**value})
            counts['candidates']+=1
            for arm in ARMS:counts[arm]+=int(value['arms'][arm])
            counts['synergistic_admissions']+=int(value['synergistic_admission'])
        print(name,family,dict(counts),flush=True)
    minimum={arm:min((len(r['features']) for r in rows if r['arms'][arm]),default=None) for arm in ARMS}
    assert minimum['legacy'] is not None and minimum['joint']<=minimum['legacy']
    for rel,sha in hashes.items():assert digest(ROOT/rel)==sha
    result={'dataset':name,'complete':True,'d':A.shape[1],'rows':rows,'counts':dict(counts),
        'minimum_feasible_count':minimum,'extra_removal':(minimum['legacy']-minimum['joint'])/A.shape[1],
        'seconds':time.perf_counter()-start,'source_hashes':hashes,'new_classifier_predictions':False,
        'validation_or_outer_observations_used':False,'goal_achieved':False,
        'scope':'existing_valid_certificate_joint_admission_training_pool_not_classifier_superiority'}
    path.write_text(json.dumps(result,separators=(',',':')),encoding='utf8');return result


def main():
    threadpool_limits(1);FOLDER.mkdir(exist_ok=True)
    scripts=('joint_admission_synergy_v1.py','run_joint_admission_synergy_v1.py','signed_classwise_dependency.py',
        'classwise_star_dependency.py','expanded_majority_guard_v1.py','diagnose_repair_bottlenecks_v7.py')
    paths=[ROOT/'scripts'/n for n in scripts]
    paths += [ROOT/'research/joint-admission-synergy-v1-protocol.md',ROOT/'data/manifest.json']
    paths += [ROOT/'src/certified_frfs'/n for n in ('core.py','majority_metric.py','margin.py',
        'validated_pool.py','datasets.py','redistributed_loss.py','classwise_dependency_certificate.py')]
    for n in NAMES:paths += [ROOT/'results/repair-relation-selector-v4'/(n+'.json'),ROOT/'data/processed'/(n+'.npz')]
    hashes={p.relative_to(ROOT).as_posix():digest(p) for p in paths}
    freeze=FOLDER/'source-freeze.json';assert not freeze.exists()
    freeze.write_text(json.dumps({'hashes':hashes,'created_unix':time.time()},indent=2),encoding='utf8')
    try:
        reports=[run(n,hashes) for n in NAMES];counts=Counter()
        for r in reports:counts.update(r['counts'])
        improved=sum(r['minimum_feasible_count']['joint']<r['minimum_feasible_count']['legacy'] for r in reports)
        removal=float(np.mean([r['extra_removal'] for r in reports]))
        summary={'complete':True,'datasets':6,'counts':dict(counts),'improved_datasets':improved,
            'mean_extra_removal':removal,'resource_gate':improved>=3 and removal>=.05,
            'dataset_rows':[{k:r[k] for k in ('dataset','d','counts','minimum_feasible_count','extra_removal','seconds')} for r in reports],
            'artifact_hashes':{n:digest(FOLDER/(n+'.json')) for n in NAMES},'goal_achieved':False,
            'new_classifier_predictions':False,'classification_superiority_established':False,
            'semantic_target':'per_class_mean_signed_dependency_decrease_not_mean_positive_object_loss',
            'limits':['Integration of existing sufficient bounds; no new theorem novelty claimed.',
                      'Only one saved inner training block per dataset, no prediction or generalization evidence.',
                      'Independent numerical replay of the combined admission remains required.']}
        (FOLDER/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8');print(json.dumps(summary,indent=2))
    except Exception:
        (FOLDER/('failure-'+str(time.time_ns())+'.json')).write_text(json.dumps({'traceback':traceback.format_exc(),
            'source_hashes':hashes,'goal_achieved':False},indent=2),encoding='utf8');raise


if __name__=='__main__':main()
