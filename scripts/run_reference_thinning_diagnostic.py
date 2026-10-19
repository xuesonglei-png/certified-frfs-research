"""Registered ALL117 training-only fixed-donor stress comparison."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_simplex_soft5_knn_pilot import ROOT,digest
from continue_soft_topk_knn_pilot import verify
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from reference_thinning_score import reference_masks,thinning_score

RESCALE=ROOT/'results/postmetric-pruning-admission'
ATOMIC=ROOT/'results/atomic-postmetric-deletion'
FOLDER=ROOT/'results/reference-thinning-diagnostic'


def endpoint_key(point,score,stress_first=True):
    hard=-round(point['hard_BA'],12)
    return ((-round(score['mean_BA'],12),hard) if stress_first else (hard,))+(point['kind'],point['receiver'] if point['receiver'] is not None else -1)


def inputs(name):
    old=json.loads((RESCALE/(name+'.json')).read_text())
    atomic=json.loads((ATOMIC/(name+'.json')).read_text())
    assert old['reference_indices']==atomic['reference_indices']
    return old,atomic


def candidate_points(old,atomic):
    assert old['id']==atomic['source_candidate_id']
    points=[{'kind':'rescale','receiver':None,'features':old['features'],
             'weights':old['weights'],'hard_BA':old['hard_training_LOO_BA'],'eligible':old['eligible']}]
    for point in atomic['fit']['checks']:
        points.append({'kind':'atomic','receiver':point['feature'],'features':point['features'],
            'weights':point['weights'],'hard_BA':point['hard_training_LOO_BA'],'eligible':point['eligible']})
    return points


def fingerprints(name):
    paths=[Path(__file__),ROOT/'scripts/reference_thinning_score.py',ROOT/'scripts/audit_reference_thinning_diagnostic.py',
        ROOT/'research/reference-thinning-diagnostic-protocol.md',RESCALE/(name+'.json'),ATOMIC/(name+'.json'),
        RESCALE/'complete-audit.json',ATOMIC/'complete-audit.json',ROOT/'data/processed'/(name+'.npz')]
    paths+=sorted((ROOT/'src/certified_frfs').glob('*.py'))
    return {p.relative_to(ROOT).as_posix():digest(p) for p in paths}


def run(name):
    hashes=fingerprints(name); output=FOLDER/(name+'.json')
    if output.exists():
        saved=json.loads(output.read_text()); assert saved['fingerprints']==hashes
        if saved['complete']: return saved
    old,atomic=inputs(name); X,y=load_dataset(name); ri=np.array(old['reference_indices'])
    A,_=preprocess_fit(X[ri],X[ri][:1]); labels=y[ri]; masks=reference_masks(len(A)); cache={}
    def scored(fs,w):
        key=(tuple(fs),tuple(w))
        if key not in cache: cache[key]=thinning_score(A,labels,fs,w,masks)
        return cache[key]
    records=[]
    for original,alternative in zip(old['fit']['checks'],atomic['records'],strict=True):
        parent=scored(original['features_before'],original['weights_before'])
        points=candidate_points(original,alternative)
        scores=[scored(p['features'],p['weights']) for p in points]
        eligible=[i for i,p in enumerate(points) if p['eligible']]
        nonworse=[i for i in eligible if scores[i]['mean_BA']>=parent['mean_BA']-1e-12]
        hard=min(eligible,key=lambda i:endpoint_key(points[i],scores[i],False)) if eligible else None
        stress=min(nonworse,key=lambda i:endpoint_key(points[i],scores[i])) if nonworse else None
        records.append({'source_candidate_id':original['id'],'parent_score':parent,
            'points':points,'scores':scores,'hard_selected_id':hard,'stress_selected_id':stress})
    assert hashes==fingerprints(name)
    saved={'dataset':name,'reference_indices':ri.tolist(),'masks':masks.tolist(),'records':records,
        'unique_metrics':len(cache),'fingerprints':hashes,'complete':True,'query_labels_used':False,
        'goal_achieved':False,'scope':'fixed_training_stress_diagnostic_not_prediction_or_path'}
    output.write_text(json.dumps(saved,separators=(',',':')),encoding='utf-8')
    print(name,'training thinning complete',len(records),'donors',len(cache),'unique metrics',flush=True)
    return saved


def summarize(reports):
    rows=[]
    for report in reports:
        records=report['records']; viable=[r for r in records if r['hard_selected_id'] is not None]
        accepted=[r for r in records if r['stress_selected_id'] is not None]
        rows.append({'dataset':report['dataset'],'candidates':len(records),'original_joint_eligible':len(viable),
            'stress_nonworse_eligible':len(accepted),
            'hard_selection_stress_worse':sum(r['scores'][r['hard_selected_id']]['mean_BA']<r['parent_score']['mean_BA']-1e-12 for r in viable),
            'ranking_changes':sum(r['stress_selected_id']!=r['hard_selected_id'] for r in accepted),
            'mean_stress_delta_pp':float(np.mean([100*(r['scores'][r['stress_selected_id']]['mean_BA']-r['parent_score']['mean_BA']) for r in accepted])) if accepted else None})
    counts={k:sum(r[k] for r in rows) for k in ('candidates','original_joint_eligible','stress_nonworse_eligible','hard_selection_stress_worse','ranking_changes')}
    return {'rows':rows,'counts':counts,'datasets':len(reports),'goal_achieved':False,
        'independent_audit_required':True,'scope':'training_diagnostic_not_validation_or_superiority'}


def main():
    threadpool_limits(1); FOLDER.mkdir(exist_ok=True)
    ra,aa=verify(RESCALE),verify(ATOMIC); names=sorted(ra['artifact_hashes']); assert len(names)==15
    assert set(names)==set(aa['artifact_hashes'])
    for name in names:
        assert digest(RESCALE/(name+'.json'))==ra['artifact_hashes'][name]
        assert digest(ATOMIC/(name+'.json'))==aa['artifact_hashes'][name]
    frozen=json.loads((ROOT/'research/reference-thinning-diagnostic-source-freeze.json').read_text())
    for rel,sha in frozen['fingerprints'].items(): assert digest(ROOT/rel)==sha
    reports=[run(n) for n in names]; result=summarize(reports); assert result['counts']['candidates']==117
    (FOLDER/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)


if __name__=='__main__':main()
