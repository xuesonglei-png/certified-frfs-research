"""ALL86 audited supports; finite training-only hard5 segment refinement."""
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from run_hardaware_guarded_soft5_admission import ROOT,FOLDER as PARENT,digest
from continue_soft_topk_knn_pilot import verify
from guarded_hard5_segment_search import fit_guarded_segment
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit

FOLDER=ROOT/'results/guarded-hard5-segment-admission'


def fingerprints(name):
    files=[Path(__file__),ROOT/'scripts/guarded_hard5_segment_search.py',ROOT/'scripts/segment_hard5_candidates.py',
        ROOT/'scripts/hardaware_guarded_soft5_descent.py',ROOT/'scripts/audit_guarded_hard5_segment_admission.py',
        ROOT/'research/guarded-hard5-segment-admission-protocol.md',PARENT/(name+'.json'),PARENT/'complete-audit.json',
        ROOT/'data/processed'/(name+'.npz')]
    files+=sorted((ROOT/'src/certified_frfs').glob('*.py'))
    return {p.relative_to(ROOT).as_posix():digest(p) for p in files}


def run(name):
    threadpool_limits(1); FOLDER.mkdir(exist_ok=True)
    audit=verify(PARENT); assert audit['artifact_hashes'][name]==digest(PARENT/(name+'.json'))
    parent=json.loads((PARENT/(name+'.json')).read_text())
    for rel,sha in parent['fingerprints'].items(): assert digest(ROOT/rel)==sha
    hashes=fingerprints(name); out=FOLDER/(name+'.json')
    if out.exists():
        saved=json.loads(out.read_text()); assert saved['fingerprints']==hashes
        if saved['complete']: return saved
    X,y=load_dataset(name); ri=np.array(parent['reference_indices']); labels=y[ri]
    A,_=preprocess_fit(X[ri],X[ri][:1]); records=[]
    for old in parent['records']:
        assert old['mixtures'][-1]['accepted']
        fs,anchor=old['features'],np.array(old['anchor']); target=np.array(old['target'])
        w,fit=fit_guarded_segment(A,labels,fs,anchor,target,max_candidates=16,proposal_limit=96)
        assert fit['guard_reference']=='original_full' and fit['input_target_accepted']
        assert fit['final_hard_LOO_BA']>=fit['input_target_hard_LOO_BA']-1e-12
        records.append({'features':fs,'anchor':anchor.tolist(),'input_target':target.tolist(),'target':w.tolist(),'fit':fit,
            'gain_over_input_target_pp':100*(fit['final_hard_LOO_BA']-fit['input_target_hard_LOO_BA']),
            'gain_over_fixed_grid_pp':100*(fit['final_hard_LOO_BA']-fit['fixed_best_hard_LOO_BA'])})
        print(name,'support',len(records),'candidates',len(fit['checks']),
              'parent_gain_pp',records[-1]['gain_over_input_target_pp'],flush=True)
    assert hashes==fingerprints(name)
    result={'dataset':name,'reference_indices':ri.tolist(),'fingerprints':hashes,'records':records,
            'complete':True,'query_labels_or_accuracy_used':False,'goal_achieved':False,
            'scope':'all86_training_only_finite_segment_refinement_not_classification_evidence'}
    out.write_text(json.dumps(result,separators=(',',':')),encoding='utf-8'); return result


def summarize(reports):
    rows=[]
    for report in sorted(reports,key=lambda r:r['dataset']):
        records=report['records']
        rows.append({'dataset':report['dataset'],'supports':len(records),
            'improved_over_parent':sum(r['gain_over_input_target_pp']>1e-9 for r in records),
            'improved_over_fixed':sum(r['gain_over_fixed_grid_pp']>1e-9 for r in records),
            'mean_gain_over_parent_pp':float(np.mean([r['gain_over_input_target_pp'] for r in records])),
            'mean_gain_over_fixed_pp':float(np.mean([r['gain_over_fixed_grid_pp'] for r in records])),
            'candidate_checks':sum(len(r['fit']['checks']) for r in records),
            'guard_rejections':sum(not p['accepted'] for r in records for p in r['fit']['checks'])})
    result={'rows':rows,'datasets':len(rows),'supports':sum(r['supports'] for r in rows),
        'dataset_balanced_gain_over_parent_pp':float(np.mean([r['mean_gain_over_parent_pp'] for r in rows])),
        'dataset_balanced_gain_over_fixed_pp':float(np.mean([r['mean_gain_over_fixed_pp'] for r in rows])),
        'goal_achieved':False,'independent_audit_required':True,
        'scope':'all15_all86_training_diagnostic_not_population_or_predictive_superiority'}
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(result,flush=True)


if __name__=='__main__':
    FOLDER.mkdir(exist_ok=True); names=sorted(verify(PARENT)['artifact_hashes'])
    with ProcessPoolExecutor(max_workers=1) as executor:
        reports=[f.result() for f in as_completed([executor.submit(run,name) for name in names])]
    summarize(reports)
