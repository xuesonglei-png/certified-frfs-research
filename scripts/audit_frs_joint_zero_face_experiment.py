"""Independent all15 joint elimination geometry, trajectories and summaries."""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_frs_joint_zero_face_experiment import ROOT,SOURCE,FOLDER,frozen,prerequisites,digest
from audit_frs_coordinate_minimum import reconstruct_geometry
from audit_frs_joint_zero_face import check_elimination
from independent_frs_pair_transition_cell import scalar_cell_evaluation as pair_value
from independent_frs_one_sided_stored import scalar_cell_evaluation as one_value
from audit_forest_compact_transfer import equivalent
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit

MODES=('majority_only','stable_cell_only','one_sided','pair_transition')


def independent_statistics(saved):
    p,d=len(saved['original_support']),saved['d']
    row={'dataset':saved['dataset'],'available':saved['available'],'original_support_size':p,
        'd':d,'fit_seconds':saved['fit_seconds'],'modes':{}}
    for mode in MODES:
        if not saved['available']:
            row['modes'][mode]=None;continue
        r=saved['eliminations'][mode];size=int(np.count_nonzero(np.asarray(r['target'])));counts=Counter(s['status'] for s in r['steps'])
        value=(one_value if mode=='one_sided' else pair_value)(saved['geometry']['cell'],r['target']) if mode in ('one_sided','pair_transition') else None
        row['modes'][mode]={'attempted_faces':len(r['steps']),'accepted_faces':sum(s['accepted'] for s in r['steps']),
            'status_counts':dict(sorted(counts.items())),'exact_joint_deletions':p-size,'final_exact_support_size':size,
            'additional_attribute_removal_fraction':(p-size)/p,'total_attribute_removal_fraction':1-size/d,
            'class_upper_bounds':None if value is None else value['class_upper_bounds'],
            'FRS_sufficient_region_feasible':None if value is None else value['feasible']}
    return row


def audit(name):
    threadpool_limits(1);parent=prerequisites();hashes=frozen();counts=Counter(datasets=1)
    path=FOLDER/(name+'.json');saved=json.loads(path.read_text());source_path=SOURCE/(name+'.json');source=json.loads(source_path.read_text())
    assert saved['complete'] and saved['fingerprints']==hashes and saved['fold']==0
    assert not saved['classification_scored'] and not saved['goal_achieved']
    assert saved['source_sha256']==parent['artifact_hashes'][name]==digest(source_path)
    assert saved['available']==source['strict_cell_available']
    for k in ('reference_indices','training_indices','original_support','original_anchor','d'):
        equivalent(saved[k],source[k])
    X,y=load_dataset(name);ri,ca=np.asarray(saved['reference_indices']),np.asarray(saved['training_indices'])
    A,_=preprocess_fit(X[ri][ca],X[ri][ca])
    if saved['available']:
        equivalent(saved['geometry'],source['diagnostic']['geometry'])
        C,rhs,SC,SR=reconstruct_geometry(A,y[ri][ca],saved['geometry'],counts)
        assert tuple(saved['eliminations'])==MODES
        for mode in MODES:
            check_elimination(saved['geometry'],C,rhs,SC,SR,mode,saved['eliminations'][mode],counts)
    else:
        assert saved['geometry'] is None and saved['eliminations'] is None
    print(name,'independent joint-zero-face/primal-dual/greedy trajectory audit',dict(counts),flush=True)
    return {'dataset':name,'row':independent_statistics(saved),'counts':dict(counts),'sha256':digest(path)}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset');args=p.parse_args()
    if args.dataset:
        report=audit(args.dataset)
        (FOLDER/(args.dataset+'.smoke-audit.json')).write_text(json.dumps({'passed':True,**report},indent=2),encoding='utf-8')
    else:
        summary=json.loads((FOLDER/'complete.json').read_text());reports=[];counts=Counter();names=sorted(prerequisites()['artifact_hashes'])
        assert [r['dataset'] for r in summary['rows']]==names
        for name in names:
            r=audit(name);reports.append(r);counts.update(r['counts'])
        rows=[r['row'] for r in reports];equivalent(rows,summary['rows'])
        sums={m:{k:sum(r['modes'][m][k] for r in rows if r['available']) for k in summary['counts'][m]} for m in MODES}
        means={m:{k:float(np.mean([r['modes'][m][k] for r in rows if r['available']])) for k in summary['means'][m]} for m in MODES}
        equivalent(sums,summary['counts']);equivalent(means,summary['means'])
        assert counts['datasets']==summary['datasets']==15
        assert counts['joint_elimination_regimes']==4*sum(r['available'] for r in rows)
        hashes={r['dataset']:r['sha256'] for r in reports};equivalent(hashes,summary['artifact_hashes'])
        out={'passed':True,'counts':dict(counts),'artifact_hashes':hashes,'summary_sha256':digest(FOLDER/'complete.json'),
            'auditor_sha256':digest(Path(__file__)),'classification_scored':False,'goal_achieved':False}
        (FOLDER/'complete-audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2),flush=True)
