"""ALL15 train-only outer metric certificate feasibility diagnostic.

Uses already independently audited FGS120 outer metrics and original training
references only. It performs no refit, projection, validation endpoint selection
or new prediction. A strong baseline may be accurate yet outside this method's
original certificate cell; that is a tradeoff diagnostic, not a baseline failure.
"""
import json
import time
from pathlib import Path
from collections import Counter
import numpy as np
from threadpoolctl import threadpool_limits
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from frs_pair_transition_cell import build_class_cell,evaluate_class_cell
from independent_frs_pair_transition_cell import scalar_cell_evaluation
from continue_soft_topk_knn_pilot import verify
from run_simplex_soft5_knn_pilot import ROOT,digest

SOURCE=ROOT/'results/fgs120-atomic-count-controls'
PARENT=ROOT/'results/frs-joint-zero-face'
FREEZE=ROOT/'research/baseline-certificate-tradeoff-v1-source-freeze.json'
OUT=ROOT/'research/baseline-certificate-tradeoff-v1.json'


def main():
    threadpool_limits(1);assert not FREEZE.exists() and not OUT.exists()
    source_audit=verify(SOURCE);parent_audit=verify(PARENT);assert len(source_audit['artifact_hashes'])==15
    paths=[ROOT/'scripts/diagnose_baseline_certificate_tradeoff_v1.py',ROOT/'scripts/frs_pair_transition_cell.py',
        ROOT/'scripts/independent_frs_pair_transition_cell.py',SOURCE/'complete.json',SOURCE/'complete-audit.json',
        PARENT/'complete.json',PARENT/'complete-audit.json']
    hashes={p.relative_to(ROOT).as_posix():digest(p) for p in paths}
    FREEZE.write_text(json.dumps({'created_unix':time.time(),'fingerprints':hashes,
        'source_artifact_hashes':source_audit['artifact_hashes'],'parent_artifact_hashes':parent_audit['artifact_hashes'],
        'scope':'ALL15_outer_FGS120_metric_original_FRS_feasibility_no_new_predictions'},indent=2),encoding='utf-8')
    rows=[];counts=Counter()
    for name,sha in sorted(source_audit['artifact_hashes'].items()):
        s=json.loads((SOURCE/(name+'.json')).read_text());assert digest(SOURCE/(name+'.json'))==sha
        r=s['outer']['results']['fgs_weighted120_atomic_count'];metric=r['metric'];X,y=load_dataset(name)
        ri=np.asarray(s['reference_indices']);A,_=preprocess_fit(X[ri],X[ri][:1]);labels=y[ri]
        parent=json.loads((PARENT/(name+'.json')).read_text());full=np.asarray(parent['geometry']['cell']['full_weights'])
        fs=metric['features'];w=np.asarray(metric['weights']);mass=np.sqrt(A.shape[1]);
        row={'dataset':name,'dimension':int(A.shape[1]),'baseline_support':len(fs),
            'baseline_clean_BA':r['evaluation']['metrics']['clean_BA'],
            'baseline_query_certified_BA':r['evaluation']['metrics']['query_certified_BA'],
            'baseline_mean_perturbed_BA':r['evaluation']['metrics']['mean_perturbed_BA'],
            'baseline_mass_error':float(w.sum()-mass)}
        if abs(row['baseline_mass_error'])>1e-7:
            row.update({'production_anchor_feasible':False,'scalar_anchor_feasible':False,'status':'mass_mismatch'});rows.append(row);continue
        try:
            C,rhs,cell=build_class_cell(A,labels,fs,w,full)
            production=evaluate_class_cell(cell,w);scalar=scalar_cell_evaluation(cell,w)
            row.update({'production_anchor_feasible':bool(production['feasible'] and (not len(C) or np.all(C@w>=rhs))),
                'scalar_anchor_feasible':bool(scalar['feasible']),'production_cell_upper_bounds':production['raw_upper_bounds_with_inflation'],
                'scalar_cell_upper_bounds':scalar.get('raw_upper_bounds_with_inflation'),
                'min_linear_slack':None if not len(C) else float(np.min(C@w-rhs)),
                'status':'feasible' if production['feasible'] and scalar['feasible'] else 'certificate_cell_rejects_baseline'})
            counts['production_feasible']+=row['production_anchor_feasible'];counts['scalar_feasible']+=row['scalar_anchor_feasible']
            counts['production_scalar_agree']+=int(row['production_anchor_feasible']==row['scalar_anchor_feasible'])
        except Exception as error:
            row.update({'production_anchor_feasible':False,'scalar_anchor_feasible':False,'status':'diagnostic_error:'+type(error).__name__})
        rows.append(row);print(name,row['status'],row['baseline_clean_BA'],flush=True)
    for rel,sha in hashes.items():assert digest(ROOT/rel)==sha,rel
    eligible=[r for r in rows if r['status']!='mass_mismatch'];
    out={'passed':True,'rows':rows,'counts':dict(counts),'datasets':len(rows),'eligible_for_cell_check':len(eligible),
        'freeze_sha256':digest(FREEZE),'source_audit_sha256':digest(SOURCE/'complete-audit.json'),
        'no_new_fit_or_prediction':True,'heldout_labels_used':False,'endpoint_selected':False,
        'classification_superiority_established':False,'goal_achieved':False,
        'limits':['Baseline metrics are existing independently audited development artifacts.',
            'Certificate-cell rejection does not disprove FGS robustness under its own objective.',
            'This does not project/retrain the baseline or compare new heldout predictions.',
            'Production and scalar feasibility are necessary diagnostics, not a population guarantee.']}
    OUT.write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps({'counts':dict(counts)},indent=2),flush=True)


if __name__=='__main__':main()
