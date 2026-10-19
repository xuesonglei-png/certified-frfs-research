import numpy as np
from dsnfrs_formula_baseline import DirectedSoftNeighborhood
from recheck_dsnfrs_published_example import TABLE1,PRINTED,rational_profile


def test_exact_rational_directed_distances_reproduce_printed_memberships():
    rows=rational_profile([0])
    assert [r['rounded4'] for r in rows]==[format(float(v),'.4f') for v in PRINTED]
    # For x6, target x15 has sample variance107/250 in sepal length.
    assert rows[5]['exact_selected_distance']=='160/107'
    assert rows[5]['enemy_object']==15


def test_full_subset_does_not_silently_replace_the_compatible_singleton():
    A=np.array(TABLE1,float); y=np.repeat([1,2,3],5)
    model=DirectedSoftNeighborhood(A,y,k=5,beta=1.6,variance_ddof=1,
        zero_variance_lambda=1e-6,singleton_variance='zero')
    one=model.profile(model.distances([0]))['own_class_lower']
    full=model.profile(model.distances(range(4)))['own_class_lower']
    assert [format(v,'.4f') for v in one]==[format(float(v),'.4f') for v in PRINTED]
    assert np.max(abs(np.array(full)-np.array(PRINTED,float)))>.2
    assert TABLE1[2][3]=='.2' and TABLE1[3][3]=='.3' and TABLE1[4][3]=='.2'
