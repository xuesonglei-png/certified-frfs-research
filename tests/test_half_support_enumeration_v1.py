import sys
from pathlib import Path
import itertools
import math
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from half_support_enumeration_v1 import possible_nominal_majority,reasons_of,NominalWitnesses


def test_all_tied_neighbors_are_conservative():
    y=np.array([0,0,0,0,1,1,1]);D=np.ones((7,7));np.fill_diagonal(D,0)
    possible,count=possible_nominal_majority(D,y)
    assert possible[:4].all() and (count[:4]==3).all()
    assert not possible[4:].any()


def test_unambiguous_three_enemy_topfive_cannot_certify():
    y=np.array([0,1,1,1,0,0,0]);D=np.tile(np.arange(7,dtype=float),(7,1));D[0]=[0,.1,.2,.3,.4,.5,.6]
    possible,count=possible_nominal_majority(D,y)
    assert not possible[0] and count[0]==2


def test_pruning_threshold_has_buffer_and_majority_is_integer():
    row={'positive_global_loss':.1+5e-10,'positive_class_loss':[.1+5e-10],
         'signed_class_loss':[.09],'possible_majority_counts':[9],'required_counts':[9]}
    assert reasons_of(row)==[]
    row['possible_majority_counts']=[8]
    assert reasons_of(row)==['nominal_tie_conservative_original_protected_majority_shortfall']
    row['positive_class_loss']=[.11]
    assert 'nominal_positive_class_dependency' in reasons_of(row)


def test_lexicographic_combinations_cover_full_declared_cohort():
    expected={4:6,9:126,13:1716,7:35}
    for d,n in expected.items():
        rows=list(itertools.combinations(range(d),math.ceil(d/2)))
        assert rows==sorted(set(rows)) and len(rows)==n


def test_full_uniform_has_no_nominal_dependency_violation():
    rng=np.random.default_rng(42);A=rng.random((12,4));y=np.repeat([0,1],6)
    out=NominalWitnesses(A,y,np.ones(4)/2,np.zeros(12,bool))(range(4))
    assert out['positive_global_loss']==0 and out['positive_class_loss']==[0.,0.]
    assert out['pruning_reasons']==[]
