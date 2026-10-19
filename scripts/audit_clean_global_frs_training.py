"""Independently replay matched nominal loss and unchanged robust constraints."""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_clean_global_frs_training import ROOT,SOURCE,FOLDER,frozen,prerequisites,digest,ARMS
from audit_clean_global_training_block import check_block
from audit_forest_compact_transfer import equivalent
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit


def audit(name):
    threadpool_limits(1);parent=prerequisites();hashes=frozen();counts=Counter(datasets=1)
    path=FOLDER/(name+'.json');saved=json.loads(path.read_text());source_path=SOURCE/(name+'.json')
    source=json.loads(source_path.read_text())
    assert saved['complete'] and saved['fingerprints']==hashes and saved['fold']==0
    assert not saved['classification_scored'] and not saved['goal_achieved']
    assert saved['source_sha256']==parent['artifact_hashes'][name]==digest(source_path)
    assert saved['reference_indices']==source['reference_indices'] and saved['training_indices']==source['training_indices']
    assert saved['available_original_cell']==source['strict_cell_available']
    X,y=load_dataset(name);ri,ca=np.asarray(saved['reference_indices']),np.asarray(saved['training_indices'])
    assert saved['d']==X.shape[1];A,_=preprocess_fit(X[ri][ca],X[ri][ca])
    if source['block'] is None:
        assert saved['block'] is None
    else:
        check_block(A,y[ri][ca],source['block'],saved['block'],counts)
    block=saved['block'];row={'dataset':name,'d':saved['d'],'available_original_cell':saved['available_original_cell'],
        'available_global_region':None if block is None else block['control']['available'],
        'evaluation':None if block is None else block['evaluation']}
    print(name,'independent same starts/budgets/nominal and robust losses/LPs audited',dict(counts),flush=True)
    return {'dataset':name,'row':row,'counts':dict(counts),'sha256':digest(path)}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset');args=p.parse_args()
    if args.dataset:
        report=audit(args.dataset)
        (FOLDER/(args.dataset+'.smoke-audit.json')).write_text(json.dumps({'passed':True,**report},indent=2),encoding='utf-8')
    else:
        summary=json.loads((FOLDER/'complete.json').read_text());names=sorted(prerequisites()['artifact_hashes']);reports=[];counts=Counter()
        assert [r['dataset'] for r in summary['rows']]==names
        for name in names:
            report=audit(name);reports.append(report);counts.update(report['counts'])
        rows=[r['row'] for r in reports];equivalent(rows,summary['rows']);valid=[r['evaluation'] for r in rows if r['evaluation'] is not None]
        means={k:None if not valid else float(np.mean([r[k] for r in valid])) for k in summary['dataset_balanced_metrics']}
        failures={a:sum(r['optimizer_failures'][a] for r in valid) for a in ARMS}
        zeros={a:sum(r['exact_zero_weights'][a]>0 for r in valid) for a in ARMS}
        equivalent(means,summary['dataset_balanced_metrics']);equivalent(failures,summary['optimizer_failures']);equivalent(zeros,summary['folds_with_exact_zero_weights'])
        assert counts['datasets']==summary['datasets']==15 and summary['available_global_regions']==len(valid)
        assert counts['matched_nominal_robust_training_folds']==len(valid)
        hashes={r['dataset']:r['sha256'] for r in reports};equivalent(hashes,summary['artifact_hashes'])
        output={'passed':True,'counts':dict(counts),'artifact_hashes':hashes,'summary_sha256':digest(FOLDER/'complete.json'),
            'auditor_sha256':digest(Path(__file__)),'goal_achieved':False,'classification_scored':False}
        (FOLDER/'complete-audit.json').write_text(json.dumps(output,indent=2),encoding='utf-8');print(json.dumps(output,indent=2),flush=True)
