import numpy as np
from soft_plurality_objective_v1 import terms_and_temperature,plurality_objective,soft_to_hard_plurality_certificate
from test_dynamic_certificate_metric_v1 import table


def test_multiclass_gradient_and_label_permutation_equivariance():
    A,y=table();w=np.array([.44,.56]);terms,tau,_=terms_and_temperature(A,y,w)
    value,gradient=plurality_objective(w,terms,y,tau)
    direction=np.array([1.,-1.]);h=1e-7
    numeric=(plurality_objective(w+h*direction,terms,y,tau)[0]-plurality_objective(w-h*direction,terms,y,tau)[0])/(2*h)
    np.testing.assert_allclose(numeric,gradient@direction,atol=2e-6,rtol=2e-5)
    renamed=np.array([27,-8,5])[y];renamed_terms,renamed_tau,_=terms_and_temperature(A,renamed,w)
    # Class sorting can change the seeded groups: objective label invariance
    # is checked on the SAME groups/distances rather than assuming split invariance.
    C,same,mask,labels=terms
    renamed_terms=(C,same,mask,np.array([27,-8,5])[labels])
    other,ograd=plurality_objective(w,renamed_terms,renamed,tau)
    np.testing.assert_allclose(other,value,atol=1e-12)
    np.testing.assert_allclose(ograd,gradient,atol=1e-10)


def test_conditional_theorem_accepts_plurality_without_strict_majority():
    d=np.array([.01,.02,.03,.04,.05,.8,.9,.95]);y=np.array([0,0,1,2,3,1,2,3])
    report=soft_to_hard_plurality_certificate(d,y,0,.001)
    assert report['certified'] and np.sum(y[:5]==0)==2
    assert not soft_to_hard_plurality_certificate(d,y,1,.001)['certified']


def test_conditional_leakage_certificate_against_random_hard_votes():
    rng=np.random.default_rng(449)
    for _ in range(80):
        n=int(rng.integers(7,50));d=rng.uniform(0,1,n);y=rng.integers(0,4,n);tau=float(rng.uniform(.001,.06))
        for label in np.unique(y):
            report=soft_to_hard_plurality_certificate(d,y,label,tau)
            if report['certified']:
                neighbor=y[np.argsort(d,kind='stable')[:5]]
                counts={c:sum(neighbor==c) for c in np.unique(y)}
                assert all(counts[label]>v for c,v in counts.items() if c!=label)
