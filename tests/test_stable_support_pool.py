from certified_frfs.stable_support_pool import support_pool,select_stable_pool


def test_pool_retains_nonlocal_recent_prefix_and_deduplicates_origins():
    pool = support_pool([0,1],5,[('fgs',[3,4]),('sfrip',[4,3]),('duplicate_old',[1,0])])
    assert len(pool) == 8
    record = next(r for r in pool if r['features'] == [3,4])
    assert record['origins'] == ['fgs','sfrip']


def test_stable_selection_rejects_partition_specific_gain_and_infeasible_best():
    class Risk:
        def __init__(self,seed,errors): self.seed,self.errors = seed,errors
        def __call__(self,fs): return {'balanced_error':self.errors[tuple(fs)],'balanced_hinge':0.}
    pool = [{'features':[a],'origins':['fixture']} for a in range(4)]
    risks = [Risk(1,{(0,):.5,(1,):.1,(2,):.2,(3,):.3}),Risk(2,{(0,):.5,(1,):.6,(2,):.2,(3,):.35})]
    def candidate(fs): return {'features':fs,'certificate':{'feasible':fs != [2]}}
    result = select_stable_pool([0],pool,risks,candidate)
    assert result['features'] == [3] and result['ranked_eligible'] == [[2],[3]]
    assert len(result['attempts']) == 2 and not result['attempts'][0]['accepted']
