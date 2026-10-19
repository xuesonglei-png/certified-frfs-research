import itertools
import json
import numpy as np
import pytest
from certified_frfs.beam_utility import beam_utility_path,beam_utility_candidates
from certified_frfs.utility_search import UtilityWorkspace,utility_path,utility_candidates
from certified_frfs.core import certificate
from certified_frfs.majority_metric import MajorityTerms
from certified_frfs.adaptive_witness import adaptive_bounds


def test_beam_escapes_irreversible_deletion_order():
    class Objective:
        def __init__(self):self.cache={};self.splits=[]
        def evaluate(self,fs):
            key=tuple(fs)
            if key not in self.cache:
                score={(1,2,3):.9,(0,2,3):.8,(0,1,3):.7,(0,1,2):.6,(0,3):.99}.get(key,.1)
                self.cache[key]={'utility':score,'clean_LOO':score,'shared_noise_LOO':score}
            return self.cache[key]
    X=np.linspace(.1,.9,48).reshape(12,4);y=np.repeat([0,1],6);w=np.ones(4)
    greedy=utility_path(X,y,w,certified=False,ranking_workspace=Objective())
    rows,beam=beam_utility_candidates(X,y,w,certified=False,ranking_workspace=Objective(),width=3)
    assert json.loads(json.dumps(beam))==beam
    assert all(0 not in fs for fs in greedy['candidates'][1:])
    assert next(r['features'] for r in rows if r['config']['fraction']==.5)==[0,3]
    assert [0,3] in beam['candidates']


def test_wide_unconstrained_beam_matches_exhaustive_objective_at_each_size():
    rng=np.random.default_rng(907);X=rng.uniform(.1,.9,(24,5));y=np.repeat([0,1,2],8);w=np.ones(5)/np.sqrt(5)
    ws=UtilityWorkspace(X,y,w);path=beam_utility_path(X,y,w,width=10,certified=False)
    for size in range(1,6):
        available=[tuple(fs) for fs in path['candidates'] if len(fs)==size]
        assert set(available)==set(itertools.combinations(range(5),size))
        assert max(ws.evaluate(fs)['utility'] for fs in available)==max(ws.evaluate(fs)['utility'] for fs in itertools.combinations(range(5),size))


def test_every_retained_beam_node_satisfies_both_original_constraints():
    rng=np.random.default_rng(908);y=np.repeat([0,1],12)
    X=np.column_stack([.1+.8*y,rng.uniform(.1,.9,(24,4))]);w=np.array([1.,.02,.02,.02,.02])
    path=beam_utility_path(X,y,w,width=3,dependency_eta=.1,shared_loss=True)
    ws=MajorityTerms(X,y,weights=w);protected=ws.bounds(range(5),allowed_enemies=2)>1e-10
    for fs in path['candidates']:
        assert certificate(X,y,fs,weights=w,eps=.01,method='hybrid')['upper_bound']<=.1
        bounds=np.maximum(ws.bounds(fs,allowed_enemies=2),adaptive_bounds(X,y,fs,w))
        assert all(int(((bounds>1e-10)&protected&(y==label)).sum())>=need
                   for label,need in zip(np.unique(y),path['required']))
    for step in path['trace']:
        assert len({tuple(fs) for fs in step['accepted_beam']})==len(step['accepted_beam'])
        passing=[a['features'] for a in step['certification_attempts'] if a['feasible']]
        assert passing==step['accepted_beam']


def test_width_one_dispatch_preserves_previous_branch_and_rejects_invalid_width():
    rng=np.random.default_rng(909);X=rng.uniform(.1,.9,(18,3));y=np.repeat([0,1],9);w=np.ones(3)
    assert utility_candidates(X,y,w)==utility_candidates(X,y,w,beam_width=1)
    for width in (0,-1,1.5):
        with pytest.raises(ValueError):utility_candidates(X,y,w,beam_width=width)
    with pytest.raises(ValueError):beam_utility_path(X*20,y,w)
