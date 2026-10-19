"""Independent first-inner support reconstruction and clean/noisy query replay."""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_frs_joint_zero_face_prediction import ROOT,JOINT,NATIVE,FOLDER,frozen,prerequisites,digest,METHODS,NATIVE_ARMS,REGIMES
from independent_geometric_evaluation import replay_final
from audit_forest_compact_transfer import equivalent
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit


def reconstruct_metrics(joint,native):
    first=native['inner'][0]
    assert joint['reference_indices']==native['reference_indices'] and joint['fold']==0
    assert joint['training_indices']==first['reference_indices']
    assert not set(first['reference_indices'])&set(first['validation_indices'])
    base=first['bases']['proposed'][0]
    assert base['features']==joint['original_support'];equivalent(base['weights'],joint['original_anchor'])
    d=joint['d'];metrics={'anchor':{'features':joint['original_support'],'weights':joint['original_anchor']},
        'full_uniform':{'features':list(range(d)),'weights':[1/np.sqrt(d)]*d}}
    for mode in REGIMES:
        point=joint['eliminations'][mode]['target'] if joint['available'] else joint['original_anchor']
        active=[(a,w) for a,w in zip(joint['original_support'],point,strict=True) if w!=0.]
        metrics['joint_'+mode]={'features':[a for a,w in active],'weights':[w for a,w in active]}
        if active:
            assert all(w>0 for a,w in active)
    choices={};widths={'fgs_native120':120,'fgs_weighted_native120':1680,'sfrip_native6':6,'sfrip_weighted_native6':84}
    for method in NATIVE_ARMS:
        selection=native['selection'][method];index=selection['selected_index']
        options=first['options'][method];assert len(options)==len(selection['options'])==widths[method]
        original=options[index]['metric'];metrics[method]={'features':original['features'],'weights':original['weights']}
        choices[method]={'selected_index':index,'candidate_count':widths[method],'recipe':selection['options'][index]}
    return metrics,choices


def independent_row(saved,metrics):
    scores={m:saved['results'][m]['evaluation']['metrics'] for m in METHODS}
    sizes={m:len(metrics[m]['features']) for m in METHODS};d=saved['d']
    return {'dataset':saved['dataset'],'d':d,'metrics':scores,'actual_feature_counts':sizes,
        'unit_cost_removed_fractions':{m:(d-sizes[m])/d for m in METHODS},
        'prediction_seconds':saved['prediction_seconds'],
        'pair_minus_anchor_BA_pp':100*(scores['joint_pair_transition']['clean_BA']-scores['anchor']['clean_BA']),
        'pair_minus_one_sided_BA_pp':100*(scores['joint_pair_transition']['clean_BA']-scores['joint_one_sided']['clean_BA'])}


def audit(name):
    threadpool_limits(1);parents=prerequisites();hashes=frozen();counts=Counter(datasets=1)
    path=FOLDER/(name+'.json');saved=json.loads(path.read_text())
    assert saved['complete'] and saved['fingerprints']==hashes and saved['classification_scored']
    assert not saved['query_labels_used_for_joint_selection'] and not saved['new_outer_evaluation_performed']
    assert not saved['goal_achieved'] and not saved['classification_superiority_established']
    for p in (JOINT,NATIVE):
        assert saved['source_hashes'][p.name]==parents[p.name]['artifact_hashes'][name]==digest(p/(name+'.json'))
    joint=json.loads((JOINT/(name+'.json')).read_text());native=json.loads((NATIVE/(name+'.json')).read_text())
    metrics,choices=reconstruct_metrics(joint,native);equivalent(metrics,saved['metrics']);equivalent(choices,saved['native_choices'])
    assert saved['reference_indices']==joint['reference_indices']
    assert saved['training_indices']==native['inner'][0]['reference_indices']
    assert saved['validation_indices']==native['inner'][0]['validation_indices']
    assert tuple(saved['results'])==METHODS and saved['fold']==0 and saved['d']==joint['d']
    X,y=load_dataset(name);ri=np.asarray(saved['reference_indices']);ca=np.asarray(saved['training_indices']);cb=np.asarray(saved['validation_indices'])
    A,B=preprocess_fit(X[ri][ca],X[ri][cb])
    for method in METHODS:
        local=Counter();record={'metric':metrics[method],**saved['results'][method]}
        replay_final(A,y[ri][ca],B,y[ri][cb],record,local)
        counts['independent_inner_validation_predictions']+=local.pop('independent_outer_predictions')
        counts.update(local)
    print(name,'all10 arms independent first-inner clean/query/16perturbation replay',dict(counts),flush=True)
    return {'dataset':name,'row':independent_row(saved,metrics),'counts':dict(counts),'sha256':digest(path)}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset');args=p.parse_args()
    if args.dataset:
        report=audit(args.dataset)
        (FOLDER/(args.dataset+'.smoke-audit.json')).write_text(json.dumps({'passed':True,**report},indent=2),encoding='utf-8')
    else:
        summary=json.loads((FOLDER/'complete.json').read_text());names=sorted(prerequisites()[JOINT.name]['artifact_hashes']);reports=[];counts=Counter()
        assert [r['dataset'] for r in summary['rows']]==names
        for name in names:
            report=audit(name);reports.append(report);counts.update(report['counts'])
        rows=[r['row'] for r in reports];equivalent(rows,summary['rows'])
        means={m:{k:float(np.mean([r['metrics'][m][k] for r in rows])) for k in rows[0]['metrics'][m]} for m in METHODS}
        equivalent(means,summary['means'])
        for method,delta in summary['pair_clean_BA_differences_pp'].items():
            equivalent(float(np.mean([100*(r['metrics']['joint_pair_transition']['clean_BA']-r['metrics'][method]['clean_BA']) for r in rows])),delta)
            wins=sum(r['metrics']['joint_pair_transition']['clean_BA']-r['metrics'][method]['clean_BA']>1e-12 for r in rows)
            losses=sum(r['metrics'][method]['clean_BA']-r['metrics']['joint_pair_transition']['clean_BA']>1e-12 for r in rows)
            assert summary['pair_wins_losses'][method]=={'wins':wins,'losses':losses}
        equivalent({m:float(np.mean([r['actual_feature_counts'][m] for r in rows])) for m in METHODS},summary['mean_actual_feature_counts'])
        assert counts['datasets']==summary['datasets']==15
        assert counts['independent_inner_validation_predictions']==150 and counts['independent_noise_predictions']==2400
        hashes={r['dataset']:r['sha256'] for r in reports};equivalent(hashes,summary['artifact_hashes'])
        report={'passed':True,'counts':dict(counts),'artifact_hashes':hashes,'summary_sha256':digest(FOLDER/'complete.json'),
            'auditor_sha256':digest(Path(__file__)),'goal_achieved':False,'classification_superiority_established':False,'new_outer_evaluation_performed':False}
        (FOLDER/'complete-audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2),flush=True)
