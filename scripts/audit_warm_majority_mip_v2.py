"""Scalar validation of the supplied seed plus the original independent replay."""
import numpy as np
from audit_fixed_face_majority_milp_v2 import check_fit as check_cold,rank_report
from audit_forest_compact_transfer import equivalent


def check_fit(A,y,fit,counts):
    check_cold(A,y,fit,counts);record=fit['warm_start'];p,n,mass=len(fit['features']),len(A),fit['mass']
    assert record['solver']=='highspy' and record['threads']==1 and record['random_seed']==0
    assert record['seed_supplied_before_run'] and record['seed_is_known_feasible_partial_assignment_not_test_label_tuning']
    seed=fit['seed'];fs=fit['features'];report=fit['seed_report']  # independently validated nearest-rank tie choices above
    expected={j:float(v/mass) for j,v in enumerate(seed)}
    for i,(row,decl) in enumerate(zip(report['rows'],fit['declarations'],strict=True)):
        success=row['qualified'];expected[p+i]=float(success)
        friends=set(row['three_friends']) if success else set();enemies=set(row['two_allowed_enemies']) if success else set()
        threshold=0. if not success else max(sum(abs(A[f,a]-A[i,a])*seed[j]/mass for j,a in enumerate(fs)) for f in friends)
        expected[decl['threshold_index']]=threshold
        for f,j in zip(decl['friends'],decl['friend_selection_indices'],strict=True):expected[j]=float(f in friends)
        for e,j in zip(decl['enemies'],decl['enemy_exception_indices'],strict=True):expected[j]=float(e in enemies)
    assert record['partial_seed_indices']==sorted(expected)
    equivalent(record['partial_seed_values'],[expected[j] for j in sorted(expected)])
    counts['independent_known_feasible_seed_assignments']+=1
