import numpy as np
from certified_frfs.nested_teacher_support import inner_splits, choose_policy, FRACTIONS

def test_singletons_remain_reference_only_and_queries_cover_repeated_objects():
    y=np.array([0,0,0,1,1,1,2]);splits=inner_splits(y)
    assert len(splits)==3
    queries=[]
    for a,b in splits:
        assert 6 in a and 6 not in b and set(a).isdisjoint(b)
        assert set(y[a])=={0,1,2}
        queries.extend(b.tolist())
    assert sorted(queries)==list(range(6))

def test_policy_selection_respects_actual_fallback_size_and_fixed_tolerance():
    row={'policies':{str(f):{'clean':.8,'retained':1.} for f in FRACTIONS}}
    row['policies']['0.65']={'clean':.796,'retained':.65}
    row['policies']['0.8']={'clean':.79,'retained':.8}
    selected,trace=choose_policy([row])
    assert selected=='0.65' and '0.8' not in trace['eligible']
    assert choose_policy([row],tolerance=0)[0]=='0.5'
