"""ALL seven training sources; interval proposals versus fixed grid."""
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from run_simplex_soft5_knn_pilot import ROOT, FOLDER, digest
from certified_frfs.datasets import load_dataset
from certified_frfs.validated_pool import preprocess_fit
from certified_frfs.feasible_transport import weight_certificate
from audit_flexible_majority_repair_admission import independent_protected
from audit_classwise_repair_admission import independent_class_certificate
from audit_simplex_source_budget_diagnostic import independent_loo
from hardaware_guarded_soft5_descent import hard5_LOO_BA
from segment_hard5_candidates import segment_candidates


def main():
    threadpool_limits(1)
    path=ROOT/'research/simplex-source-budget-diagnostic.json'
    audit=ROOT/'research/simplex-source-budget-diagnostic-audit.json'
    checked=json.loads(audit.read_text()); assert checked['passed'] and checked['diagnostic_sha256']==digest(path)
    saved=json.loads(path.read_text())
    for rel,sha in saved['fingerprints'].items(): assert digest(ROOT/rel)==sha
    files=[path,audit,Path(__file__),ROOT/'scripts/segment_hard5_candidates.py',
           ROOT/'research/hard5-segment-source-diagnostic-protocol.md']
    hashes={p.relative_to(ROOT).as_posix():digest(p) for p in files}
    rows=[]
    for source in saved['rows']:
        name=source['dataset']; parent=json.loads((FOLDER/(name+'.json')).read_text())
        block=parent['inner'][source['block']]; learner=block['learners'][source['learner']]
        X,y=load_dataset(name); ri=np.array(parent['reference_indices']); ca=np.array(block['reference_indices'])
        raw,labels=X[ri][ca],y[ri][ca]; A,_=preprocess_fit(raw,raw[:1])
        fs=learner['features']; anchor=np.array(learner['anchor']); target=np.array(source['retry_fit']['weights'])
        started=time.perf_counter()
        proposals,info=segment_candidates(A[:,fs],labels,anchor,target,max_candidates=16)
        elapsed=time.perf_counter()-started
        full=np.ones(A.shape[1])/np.sqrt(A.shape[1]); protected=independent_protected(A,labels,full)
        points=[]
        for p in proposals:
            w=(1-p['t'])*anchor+p['t']*target
            cert=weight_certificate(A,labels,fs,full,w,protected=protected)
            classes=independent_class_certificate(labels,cert)
            actual=hard5_LOO_BA(A[:,fs],labels,w)
            scalar=independent_loo(A,labels,fs,w)
            assert abs(actual-scalar)<=1e-12
            points.append({**p,'weights':w.tolist(),'actual_hard_LOO_BA':actual,
                'affine_estimate_discrepancy_pp':None if p['estimated_hard_LOO_BA'] is None else 100*(actual-p['estimated_hard_LOO_BA']),
                'certificate':cert,'class_certificate':classes,
                'accepted':bool(cert['feasible'] and classes['feasible_all_classes'])})
        accepted=[p for p in points if p['accepted']]
        fixed=[p for p in accepted if p['origin']=='fixed_grid']
        best=max((p['actual_hard_LOO_BA'] for p in accepted),default=None)
        old=max((p['actual_hard_LOO_BA'] for p in fixed),default=None)
        row={'dataset':name,'block':source['block'],'learner':source['learner'],'n':len(A),'features':fs,
             'proposal_info':info,'proposal_seconds':elapsed,'points':points,
             'best_accepted_BA':best,'fixed_best_accepted_BA':old,
             'gain_over_fixed_pp':None if best is None or old is None else 100*(best-old)}
        assert old is None or best>=old-1e-12
        rows.append(row)
        print(name,source['block'],source['learner'],'intervals',info['merged_intervals'],
              'proposal_seconds',round(elapsed,3),'gain_over_fixed_pp',row['gain_over_fixed_pp'],flush=True)
    assert hashes=={p.relative_to(ROOT).as_posix():digest(p) for p in files}
    result={'rows':rows,'fingerprints':hashes,'complete':True,'goal_achieved':False,'query_labels_used':False,
        'cases':len(rows),'candidate_checks':sum(len(r['points']) for r in rows),
        'cases_improved_over_fixed':sum(r['gain_over_fixed_pp'] is not None and r['gain_over_fixed_pp']>1e-9 for r in rows),
        'new_feasible_cases':sum(r['best_accepted_BA'] is not None and r['fixed_best_accepted_BA'] is None for r in rows),
        'scope':'all_seven_training_only_segment_prototype_not_predictive_advantage'}
    (ROOT/'research/hard5-segment-source-diagnostic.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print({k:v for k,v in result.items() if k not in ('rows','fingerprints')},flush=True)


if __name__=='__main__': main()
