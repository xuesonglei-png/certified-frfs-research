"""Full-reference versus own-anchor provenance and finite candidate safety."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from guarded_hard5_segment_search import proposal_sample,fit_guarded_segment
from audit_simplex_source_budget_diagnostic import independent_loo


def test_proposal_sample_retains_tiny_classes_and_is_repeatable():
    y=np.array([0]*150+[1]*20+[2]*2+[3])
    ids=proposal_sample(y)
    np.testing.assert_array_equal(ids,proposal_sample(y))
    assert len(ids)==96 and len(set(ids))==96
    assert set(ids)<=set(range(len(y)))
    assert all(sum(y[ids]==c)>=min(2,sum(y==c)) for c in np.unique(y))


def test_segment_choice_has_full_guards_and_never_loses_to_fixed_grid():
    rng=np.random.default_rng(861); A=rng.uniform(0,1,(20,6)); y=np.repeat([0,1],10)
    A[:,0]=.1+.8*y; full=np.ones(6)/np.sqrt(6)
    target=full.copy(); target[0]+=.05; target[1]-=.05
    w,fit=fit_guarded_segment(A,y,list(range(6)),full,target)
    assert fit['guard_reference']=='original_full' and fit['original_initial_guard']['accepted']
    chosen=fit['checks'][fit['selected']]; assert chosen['accepted']
    assert chosen['hard_full_training_LOO_BA']>=fit['fixed_best_hard_LOO_BA']-1e-12
    assert fit['final_hard_LOO_BA']==independent_loo(A,y,list(range(6)),w)
    assert len(fit['checks'])<=16
    # Uninformative reduced control gets its own anchor as guard reference;
    # that safeguard is not preservation of original-full dependency.
    own=np.array([np.sqrt(6)/2]*2)
    _,control=fit_guarded_segment(A,y,[1,2],own,own)
    assert control['guard_reference']=='own_anchor' and not control['original_initial_guard']['accepted']
