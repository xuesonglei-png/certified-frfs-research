"""ALL15 matched nominal objective; preserve audited robust three-start control."""
import argparse
import json
import numpy as np
from threadpoolctl import threadpool_limits
from run_simplex_soft5_knn_pilot import ROOT,digest
from continue_soft_topk_knn_pilot import verify
from run_frs_zero_face_repair_experiment import builtin
from clean_global_frs_training_ablation import fit_block,ARMS
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit

SOURCE=ROOT/'results/sparse-global-frs-training'
FOLDER=ROOT/'results/clean-global-frs-training'
FREEZE=ROOT/'research/clean-global-frs-training-source-freeze.json'
REVIEW=ROOT/'research/sparse-global-FRS-training-review.json'


def prerequisites():
    audit=verify(SOURCE);review=json.loads(REVIEW.read_text())
    assert review['audit_sha256']==digest(SOURCE/'complete-audit.json')
    assert len(audit['artifact_hashes'])==len(review['all_dataset_rows'])==15
    assert not review['classification_scored'] and not review['goal_achieved']
    return audit


def frozen():
    hashes=json.loads(FREEZE.read_text())['fingerprints']
    for rel,sha in hashes.items():
        assert digest(ROOT/rel)==sha,rel
    return hashes


def row(saved):
    block=saved['block']
    return {'dataset':saved['dataset'],'d':saved['d'],'available_original_cell':saved['available_original_cell'],
        'available_global_region':None if block is None else block['control']['available'],
        'evaluation':None if block is None else block['evaluation']}


def run(name):
    threadpool_limits(1);audit=prerequisites();hashes=frozen();FOLDER.mkdir(exist_ok=True)
    path=FOLDER/(name+'.json');source_path=SOURCE/(name+'.json')
    assert audit['artifact_hashes'][name]==digest(source_path)
    if path.exists():
        old=json.loads(path.read_text());assert old['complete'] and old['fingerprints']==hashes
        assert old['source_sha256']==digest(source_path);return
    source=json.loads(source_path.read_text());X,y=load_dataset(name)
    ri,ca=np.asarray(source['reference_indices']),np.asarray(source['training_indices'])
    A,_=preprocess_fit(X[ri][ca],X[ri][ca])
    block=None if source['block'] is None else fit_block(A,y[ri][ca],source['block'])
    saved={'dataset':name,'d':X.shape[1],'fold':0,'reference_indices':ri.tolist(),'training_indices':ca.tolist(),
        'available_original_cell':source['strict_cell_available'],'source_sha256':digest(source_path),
        'fingerprints':hashes,'block':block,'complete':True,'goal_achieved':False,'classification_scored':False,
        'scope':'exposed_training_nominal_loss_unchanged_robust_constraints_matched_three_start_control'}
    path.write_text(json.dumps(saved,default=builtin,separators=(',',':')),encoding='utf-8')
    print(name,'matched nominal/robust training',row(saved),flush=True)


def summarize(names):
    rows=[row(json.loads((FOLDER/(n+'.json')).read_text())) for n in names]
    available=[r['evaluation'] for r in rows if r['evaluation'] is not None]
    means={k:None if not available else float(np.mean([r[k] for r in available])) for k in
        ('nominal_bound_decrease_pp','robust_bound_decrease_pp')}
    failures={a:sum(r['optimizer_failures'][a] for r in available) for a in ARMS}
    zero_counts={a:sum(r['exact_zero_weights'][a]>0 for r in available) for a in ARMS}
    out={'rows':rows,'datasets':len(names),'available_global_regions':len(available),'dataset_balanced_metrics':means,
        'optimizer_failures':failures,'folds_with_exact_zero_weights':zero_counts,
        'artifact_hashes':{n:digest(FOLDER/(n+'.json')) for n in names},'independent_audit_required':True,
        'goal_achieved':False,'classification_scored':False,
        'limits':['Both losses are finite training-table bounds, not validation/test accuracy.',
                  'Different penalized objectives cannot be compared as a common improvement.',
                  'Original box=.01 eta=.1 retention=.9 and source starts unchanged.',
                  'Unavailable regions remain in all15 rows; conditional training means are explicit.']}
    (FOLDER/'complete.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k not in ('rows','artifact_hashes')},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset');args=p.parse_args()
    if args.dataset:
        run(args.dataset)
    else:
        names=sorted(prerequisites()['artifact_hashes'])
        for name in names:
            run(name)
        summarize(names)
